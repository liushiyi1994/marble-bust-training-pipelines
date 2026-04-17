# Marble Bust LoRA Evaluation & Inference Pipeline — Build Spec

## Context

This is the companion spec to `training_pipelines_spec.md`. That doc covers training 5 LoRAs (3 Architecture A + 2 Architecture B) across different base models. This doc covers what we do with those LoRAs once training finishes: run inference on a fixed eval set, score outputs automatically, and produce a comparison scorecard.

The goal is a single command that takes a folder of trained LoRAs and produces a scorecard the team can use to decide which (model, architecture) combination to productionize.

## Goals

1. Run each trained LoRA through a fixed eval set of selfies to produce output busts.
2. Automatically score every output on four dimensions: identity preservation, eye treatment, modern artifact removal, marble texture quality.
3. Produce a per-LoRA scorecard and side-by-side visual comparison grids.
4. Keep the harness model-agnostic: adding a 6th LoRA later should require a new config file, not new code.
5. Runnable on RunPod (primary) or any Linux machine with a GPU.

## Scope

In scope:
- Inference adapters for Architecture A (style LoRA + reference) and Architecture B (edit model with LoRA)
- Automated scoring: ArcFace similarity, MediaPipe eye analysis, VLM-based artifact/marble checks
- Scorecard generation: per-LoRA metrics, visual comparison grids, summary report
- Integration with training outputs: auto-discover LoRAs from an S3 bucket or local directory

Out of scope:
- Training (covered in training spec)
- Production serving pipeline
- Human-in-the-loop annotation UI (we'll review the auto-generated grids manually)
- The retry / quality gate logic from the memo (that's production, not eval)

## The Eval Set

Fixed eval set of **8 selfies**, covering:
- Gender: 4 male, 4 female
- Ethnicity: diverse (at least 4 different skin tones represented)
- Accessories: 2 with glasses, 1 with earrings, 1 with headphones, 4 plain
- Age range: at least 2 under 25, at least 2 over 40

Eval set lives in the shared network volume at `/workspace/shared/eval_set/v1/` alongside a manifest describing the demographics and accessories of each selfie. **Use the same 8 selfies across every LoRA, every seed, every run.** Do not regenerate the eval set between comparisons.

For each selfie, generate **3 outputs** per LoRA at different random seeds (seeds 42, 123, 777 — fixed and constant across LoRAs). Total outputs per LoRA: 8 selfies × 3 seeds = 24 images.

For 5 LoRAs: 120 outputs total per eval run.

## Repository Layout

Build this alongside (or in the same repo as) the training pipelines:

```
marble-bust-eval/
├── README.md
├── pyproject.toml or requirements.txt
├── Dockerfile                       # Builds the eval container
├── configs/
│   ├── eval_defaults.yaml           # Shared eval settings (eval set path, seeds, scoring thresholds)
│   ├── lora_klein_4b.yaml           # Per-LoRA inference config (Arch A)
│   ├── lora_flux2_dev.yaml          # Arch A
│   ├── lora_qwen_image.yaml         # Arch A
│   ├── lora_qwen_edit_2511.yaml     # Arch B
│   └── lora_kontext_dev.yaml        # Arch B
├── adapters/
│   ├── __init__.py
│   ├── base.py                      # Abstract InferenceAdapter interface
│   ├── arch_a_adapter.py            # Handles (prompt, reference_image) signature
│   └── arch_b_adapter.py            # Handles (input_image, prompt) signature
├── scoring/
│   ├── __init__.py
│   ├── arcface_scorer.py            # Identity similarity
│   ├── eye_scorer.py                # MediaPipe eye region analysis
│   ├── vlm_scorer.py                # Gemini Flash / GPT-4o-mini for artifact + marble checks
│   └── aggregate.py                 # Combines per-image scores into per-LoRA summary
├── prompts/
│   ├── arch_a_prompts.py            # Prompt templates for generation models
│   └── arch_b_prompts.py            # Instruction templates for edit models
├── output/
│   ├── scorecard.py                 # Produces markdown + CSV scorecard
│   └── grid_builder.py              # Side-by-side visual comparison grids
├── runpod/
│   ├── launch_eval.sh               # Single command: launches pod, runs eval, uploads results
│   └── setup_pod.sh
└── scripts/
    ├── run_eval.sh                  # Main entry: ./run_eval.sh [--loras=all]
    ├── run_inference_only.sh        # Just generate outputs, skip scoring
    └── run_scoring_only.sh          # Score pre-generated outputs (useful for re-scoring)
```

## Inference Adapters

Both architectures produce the same logical output (a marble bust given a selfie), but the model call signatures differ. Adapters normalize this.

### Architecture A Adapter

For generation models with multi-reference: FLUX.2 Klein 4B, FLUX.2 Dev, Qwen-Image (2512 base).

```python
class ArchAAdapter(InferenceAdapter):
    def infer(self, selfie: PIL.Image, persona: str, seed: int) -> PIL.Image:
        prompt = build_arch_a_prompt(persona, trigger_word=self.config.trigger_word)
        output = self.pipe(
            prompt=prompt,
            reference_images=[selfie],
            reference_strength=self.config.reference_strength,  # tunable per model
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]
        return output
```

### Architecture B Adapter

For edit models: Qwen-Image-Edit-2511, FLUX.1 Kontext Dev.

```python
class ArchBAdapter(InferenceAdapter):
    def infer(self, selfie: PIL.Image, persona: str, seed: int) -> PIL.Image:
        prompt = build_arch_b_prompt(persona, trigger_word=self.config.trigger_word)
        output = self.pipe(
            image=selfie,
            prompt=prompt,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]
        return output
```

Both adapters return a uniform `(selfie_input, bust_output)` pair downstream. Scoring code does not care which architecture produced the output.

### Persona selection

For each selfie, assign a persona **deterministically** based on the selfie index + detected gender (via the manifest). Same persona for the same selfie across every LoRA — we're comparing LoRAs, not personas. Do not randomize persona assignment during eval, it adds noise.

Recommended: hash(selfie_id) modulo gender-appropriate persona list.

## Scoring

Implement the four checks from the memo's Section 4 quality gate. Scoring runs after all inference is complete — do not mix inference and scoring in the same loop.

### 1. Identity Preservation (ArcFace)

Compute ArcFace cosine similarity between:
- Face region of the input selfie (after face detection crop)
- Face region of the output bust (same detection crop)

Use `insightface` library with `buffalo_l` model.

Output: raw cosine similarity score `[0, 1]` per output image. Higher is more similar.

**Important:** ArcFace was trained on photorealistic faces. Its behavior on marble-textured faces is not calibrated. Log the raw score — do not threshold into pass/fail during eval. Use it as a relative comparison signal across LoRAs.

### 2. Eye Treatment

Use MediaPipe Face Mesh to detect eye landmarks on the output bust. For each eye:
- Extract the iris region using landmarks 468-472 (right) and 473-477 (left)
- Compute mean RGB brightness in that region
- Compute color variance (standard deviation of RGB values)

Pass criteria:
- Mean brightness > 200 (near-white)
- Color variance < 15 (no dark iris, no colored pupil)

Output: per-image pass/fail + raw brightness + raw variance. Aggregate per-LoRA as % pass rate.

### 3. Modern Artifacts (VLM)

Use a VLM to check for accessories/objects that shouldn't be there. Recommended: Gemini Flash (cheap, good for this).

Prompt:
```
Look at this image of a marble statue bust. List any modern objects visible
(glasses, earbuds, headphones, modern jewelry, watches, digital devices,
modern clothing). Respond with ONLY a Python list, e.g. ["glasses"] or [].
No other text.
```

Output: pass = empty list returned. Fail = anything listed. Aggregate per-LoRA as % pass rate.

Cost: ~$0.002 per image. Budget for 5 LoRAs × 24 outputs = 120 calls = ~$0.24 per full eval run.

### 4. Marble Texture (VLM)

Same VLM, different prompt:
```
Is the subject of this image a marble statue, or a person with grey-tinted skin?
Respond with only one word: "marble" or "person".
```

Output: pass = "marble". Aggregate per-LoRA as % pass rate.

### Scoring Infrastructure

All scoring runs locally on CPU (or a single GPU for ArcFace) — no need for a large inference pod. Can run on the same pod as inference after inference completes, or on a separate cheaper CPU-only pod.

Scores are saved as a flat CSV: one row per output image, columns for every metric.

## Output: Scorecard

Produce two artifacts at the end of every eval run:

### 1. `scorecard.md` — per-LoRA summary

```markdown
# Marble Bust LoRA Evaluation — Run 2026-04-17

Eval set: v1 (8 selfies × 3 seeds = 24 outputs per LoRA)

| LoRA | Arch | Identity (mean ArcFace) | Eye Pass % | Artifact Pass % | Marble Pass % | Overall |
|---|---|---|---|---|---|---|
| klein_4b | A | 0.42 | 75% | 88% | 92% | ... |
| flux2_dev | A | 0.51 | 83% | 92% | 96% | ... |
| qwen_image | A | 0.48 | 71% | 79% | 88% | ... |
| qwen_edit_2511 | B | 0.58 | 79% | 83% | 92% | ... |
| kontext_dev | B | 0.54 | 67% | 75% | 85% | ... |

## Notes
- Identity scores are not directly comparable across architectures — Arch B
  models include the selfie in their conditioning pathway...
- Eye pass rate is the strongest discriminator; failures concentrate in
  selfies with heavy eye makeup in the source...
```

### 2. Visual comparison grids

For each of the 8 eval selfies, produce one grid image:

```
[ Input Selfie ] | [ klein_4b output ] | [ flux2_dev output ] | [ qwen_image output ] | [ qwen_edit_2511 output ] | [ kontext_dev output ]
```

Using seed 42 only for the grid (pick one seed so it fits on a page). Label each column with the LoRA name. Save as high-res JPG.

Also produce a "failure gallery" grid showing the 10 worst outputs by overall score across all LoRAs. This is where humans look for systematic failure modes.

## Eval Pipeline Flow

Single command: `bash scripts/run_eval.sh`

1. Read `configs/eval_defaults.yaml` → eval set path, seeds, scoring thresholds
2. Discover trained LoRAs: either from `s3://marble-bust-loras/` or a local dir. For each LoRA, read its corresponding config from `configs/lora_*.yaml`.
3. For each LoRA:
   - Load base model + LoRA weights via the appropriate adapter (A or B)
   - Run inference on 8 selfies × 3 seeds = 24 outputs
   - Save outputs to `runs/{run_id}/{lora_name}/seed_{n}/{selfie_id}.jpg`
   - Unload model, clear GPU memory
4. Once all inference is complete, run scoring:
   - ArcFace on every output
   - MediaPipe eye analysis on every output
   - VLM artifact check on every output
   - VLM marble check on every output
   - Write results to `runs/{run_id}/scores.csv`
5. Build scorecard:
   - Aggregate `scores.csv` → `runs/{run_id}/scorecard.md`
   - Build grids → `runs/{run_id}/grids/*.jpg`
   - Build failure gallery → `runs/{run_id}/failure_gallery.jpg`
6. Upload `runs/{run_id}/` to S3 for the team to review.

## Inference GPU Strategy

All inference can run on a **single pod sequentially across LoRAs** — this is simpler than coordinating 5 parallel pods and the eval is small enough that sequential completes in under an hour.

Target pod: **A100 80GB**. This is the smallest single GPU that can load all 5 base models without quantization gymnastics. Sequential inference over 5 LoRAs × 24 outputs:

| LoRA | Base model | Time per image | Total for 24 |
|---|---|---|---|
| klein_4b | FLUX.2 Klein 4B | ~1s | ~25s |
| flux2_dev | FLUX.2 Dev 32B | ~8s | ~3.5 min |
| qwen_image | Qwen-Image 20B | ~6s | ~2.5 min |
| qwen_edit_2511 | Qwen-Image-Edit 20B | ~6s | ~2.5 min |
| kontext_dev | Kontext Dev 12B | ~4s | ~1.5 min |

Total inference wall-clock: ~10 minutes + model loading overhead (~2 min per model swap) = ~20 minutes total on one A100.

**Cost per eval run:** ~$0.60 GPU time + ~$0.30 VLM calls = **under $1 per full eval run**.

Parallel mode (optional): if the team wants faster turnaround, a `--parallel` flag can split inference across 2–3 pods. Skip this in v1 unless explicitly requested.

## Scoring Reliability

Two caveats worth building into the scorecard output so we don't over-interpret:

1. **ArcFace on marble faces is uncalibrated.** We've never benchmarked it. Report raw scores and rank LoRAs relative to each other, but don't present an absolute "identity preservation score." The scorecard should say this in a footnote.

2. **Small eval set.** 24 images per LoRA is enough for coarse ranking but not for fine differences. If two LoRAs are within ~5% on pass rates, treat them as tied. The scorecard should flag close calls (within 5%) as ambiguous.

## RunPod Integration Requirements

Same pattern as the training spec — a single command spins up a pod, runs the eval, uploads results, tears down the pod.

```
bash runpod/launch_eval.sh --run-id=2026-04-17-bakeoff-1
```

Required env vars:
- `HF_TOKEN` — for gated model downloads
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — for LoRA/output storage
- `GEMINI_API_KEY` or `OPENAI_API_KEY` — for VLM scoring (pick one, default to Gemini Flash for cost)
- `RUNPOD_API_KEY` — for the launch script

Network volume `/workspace/shared` contains:
- `eval_set/v1/` — the 8 selfies and manifest
- (Optionally) cached base model weights to avoid re-downloading

## Design Principles

- **Same comparison across all LoRAs.** Same 8 selfies, same 3 seeds, same personas, same scoring. Do not let eval variance contaminate the comparison.
- **Auto-scoring gets you 80% of the signal.** Human review is for the top 2–3 LoRAs on the final grids, not for scoring 120 images by hand.
- **Fail loud on missing data.** If a LoRA's config is missing or its weights can't be loaded, skip it and note it in the scorecard — don't silently drop it.
- **Reproducible.** Given the same LoRAs and eval set, two runs of `run_eval.sh` should produce the same scores (modulo VLM variance, which is unavoidable).
- **Sequential by default.** Parallelism is an optimization, not a feature. Ship the sequential version first.

## Open Questions for the Agent to Flag

1. If a model's inference API in diffusers differs from the adapter signatures above (e.g., Kontext Dev LoRA loading has quirks), propose the correct API and wait for confirmation.
2. If VLM scoring is unreliable (Gemini Flash returns inconsistent formats), propose a validation/retry strategy before falling back to a different VLM.
3. If a LoRA fails to load (incompatible diffusers version, rank mismatch, etc.), document the failure mode so we can file it back against the training pipeline.

## Non-Goals

- Not building a production quality gate (the memo's Section 4 spec is for production, not eval).
- Not building an annotation UI — the team reviews the auto-generated grids directly.
- Not optimizing inference throughput. Eval runs are infrequent.
- Not benchmarking inference latency or cost per LoRA here — that's a production infra concern, handled separately.

## Integration With Training Pipelines

The eval harness expects LoRAs produced by the training pipelines to land in a discoverable location (default: `s3://marble-bust-loras/{pipeline_name}/{lora_name}.safetensors`). If the training spec's output schema changes, update the eval configs to match.

One eval config per trained LoRA. Keeping these in sync is manual for now (5 LoRAs is small); if we add more, consider auto-generating eval configs from training configs.
