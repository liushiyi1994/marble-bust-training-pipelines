from pathlib import Path

import pytest

from core.trainer_versions import TRAINERS
from scripts import bootstrap_trainers as bootstrap


def test_ai_toolkit_commit_is_pinned():
    assert TRAINERS["ai_toolkit"]["commit"] == "a513a1583e64cffad0ef5cd63b55ff3a5a4c6f99"


def test_diffsynth_commit_is_pinned():
    assert TRAINERS["diffsynth"]["commit"] == "079e51c9f3f296bbe636aa74448a7e3637278232"


@pytest.mark.parametrize(
    "name, repo, directory",
    [
        ("ai_toolkit", "https://github.com/ostris/ai-toolkit.git", ".vendor/ai-toolkit"),
        ("diffsynth", "https://github.com/modelscope/DiffSynth-Studio.git", ".vendor/DiffSynth-Studio"),
    ],
)
def test_trainer_repo_and_directory_are_pinned(name: str, repo: str, directory: str):
    assert TRAINERS[name]["repo"] == repo
    assert TRAINERS[name]["directory"] == directory


def test_ensure_checkout_resolves_paths_against_repo_root(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    recorded = []
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setitem(TRAINERS["ai_toolkit"], "directory", ".vendor/ai-toolkit")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, check: recorded.append(cmd))

    bootstrap.ensure_checkout("ai_toolkit")

    assert recorded == [
        ["git", "clone", TRAINERS["ai_toolkit"]["repo"], str(root / ".vendor/ai-toolkit")],
        ["git", "-C", str(root / ".vendor/ai-toolkit"), "fetch", "--all"],
        ["git", "-C", str(root / ".vendor/ai-toolkit"), "checkout", TRAINERS["ai_toolkit"]["commit"]],
    ]


def test_ensure_checkout_skips_clone_for_existing_git_repo(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    checkout = root / ".vendor" / "ai-toolkit"
    (checkout / ".git").mkdir(parents=True)
    recorded = []
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setitem(TRAINERS["ai_toolkit"], "directory", ".vendor/ai-toolkit")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, check: recorded.append(cmd))

    bootstrap.ensure_checkout("ai_toolkit")

    assert recorded == [
        ["git", "-C", str(checkout), "fetch", "--all"],
        ["git", "-C", str(checkout), "checkout", TRAINERS["ai_toolkit"]["commit"]],
    ]


def test_ensure_checkout_rejects_non_git_directory(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    checkout = root / ".vendor" / "ai-toolkit"
    checkout.mkdir(parents=True)
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setitem(TRAINERS["ai_toolkit"], "directory", ".vendor/ai-toolkit")

    with pytest.raises(RuntimeError, match="is not a git repository"):
        bootstrap.ensure_checkout("ai_toolkit")


def test_main_help_is_safe(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(bootstrap, "ensure_checkout", lambda name, dry_run=False: called.append((name, dry_run)))

    with pytest.raises(SystemExit) as excinfo:
        bootstrap.main(["--help"])

    assert excinfo.value.code == 0
    assert called == []
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_main_dry_run_produces_no_side_effects(monkeypatch, capsys, tmp_path):
    recorded = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: recorded.append((args, kwargs)))
    root = tmp_path / "repo"
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setitem(TRAINERS["ai_toolkit"], "directory", ".vendor/ai-toolkit")

    bootstrap.main(["--trainer", "ai_toolkit", "--dry-run"])

    assert recorded == []
    assert not root.exists()
    assert not (root / ".vendor").exists()
    assert not (root / ".vendor" / "ai-toolkit").exists()
    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower()
