from __future__ import annotations

try:
    from avatar_gs.smpl.stage1_vibe import TranslationConfig, fit_smpl_sequence_stage1
except Exception as exc:
    _STAGE1_VIBE_IMPORT_ERROR = exc

    class TranslationConfig:  # compatibility placeholder
        pass

    def fit_smpl_sequence_stage1(*args, **kwargs):
        raise RuntimeError(
            "stage1_vibe backend is unavailable in this repo. "
            "Please use `python -m avatar_gs.cli.stage1_fit_wham_pymaf ...` for Stage1."
        ) from _STAGE1_VIBE_IMPORT_ERROR

__all__ = ["fit_smpl_sequence_stage1", "TranslationConfig"]
