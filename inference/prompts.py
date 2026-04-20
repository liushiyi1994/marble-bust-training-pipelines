from __future__ import annotations

from core.config_schema import PipelineConfig


def _format_trigger_word(trigger_word: str) -> str:
    cleaned = trigger_word.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return cleaned
    return f"<{cleaned}>"


def build_inference_prompt(cfg: PipelineConfig, *, prompt: str | None, persona: str | None) -> str:
    if prompt:
        return prompt

    trigger = _format_trigger_word(cfg.training.trigger_word)
    persona_text = persona.strip() if persona else "timeless classical portrait"

    if cfg.architecture == "A":
        return (
            f"a {trigger} marble statue bust of a {persona_text}, white stone eyes, "
            "neutral expression, dark background with amber ember particles"
        )

    return (
        f"transform into a {trigger} marble statue bust, {persona_text} persona, "
        "white stone eyes, preserve facial bone structure and identity"
    )
