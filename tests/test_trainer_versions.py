from core.trainer_versions import TRAINERS


def test_ai_toolkit_commit_is_pinned():
    assert TRAINERS["ai_toolkit"]["commit"] == "a513a1583e64cffad0ef5cd63b55ff3a5a4c6f99"


def test_diffsynth_commit_is_pinned():
    assert TRAINERS["diffsynth"]["commit"] == "079e51c9f3f296bbe636aa74448a7e3637278232"
