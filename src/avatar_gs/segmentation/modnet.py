from __future__ import annotations

from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_MODNET_RUNTIME: dict[tuple[str, str], "_MODNetRuntime"] = {}
_MODNET_LOCK = threading.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _largest_component_by_overlap(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None, min_area: int = 128) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m * 255).astype(np.uint8)
    best_i = -1
    best_score = -1.0
    ref = None
    if ref_mask_u8 is not None:
        ref = (ref_mask_u8 > 0)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(max(1, min_area)):
            continue
        comp = (labels == i)
        if ref is not None:
            inter = float(np.logical_and(comp, ref).sum())
            cover = inter / float(max(area, 1))
            score = inter + 0.2 * cover
        else:
            score = float(area)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i < 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[labels == best_i] = 255
    return out


def _clean_mask(mask_u8: np.ndarray, ref_mask_u8: np.ndarray | None = None) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return _largest_component_by_overlap(m, ref_mask_u8, min_area=128)


def resolve_modnet_ckpt_path(ckpt_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if ckpt_path is not None and str(ckpt_path).strip() != "":
        p = Path(ckpt_path)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        candidates.append(p)
    for env_name in ("AVATAR_GS_MODNET_CKPT", "MODNET_CKPT"):
        raw = None
        try:
            raw = __import__("os").environ.get(env_name)
        except Exception:
            raw = None
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (_project_root() / p).resolve()
            candidates.append(p)
    pretrained_dir = _project_root() / "third_party" / "MODNet" / "pretrained"
    candidates.extend(
        [
            pretrained_dir / "modnet_photographic_portrait_matting.ckpt",
            pretrained_dir / "modnet_webcam_portrait_matting.ckpt",
        ]
    )
    candidates.extend(sorted(pretrained_dir.glob("*.ckpt")))
    candidates.extend(sorted(pretrained_dir.glob("*.pt")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    default_path = pretrained_dir / "modnet_photographic_portrait_matting.ckpt"
    raise FileNotFoundError(
        "MODNet checkpoint not found. Expected one of: "
        f"{default_path}. Place the ckpt there or set AVATAR_GS_MODNET_CKPT / MODNET_CKPT."
    )


def _load_modnet_class():
    modnet_root = _project_root() / "third_party" / "MODNet"
    modnet_root_str = str(modnet_root)
    if modnet_root_str not in sys.path:
        sys.path.insert(0, modnet_root_str)
    from src.models.modnet import MODNet  # type: ignore

    return MODNet


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor], want_module_prefix: bool) -> dict[str, torch.Tensor]:
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if has_module_prefix == want_module_prefix:
        return state_dict
    if want_module_prefix:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


class _MODNetRuntime:
    def __init__(self, ckpt_path: Path, device: torch.device) -> None:
        MODNet = _load_modnet_class()
        model = MODNet(backbone_pretrained=False)
        model = nn.DataParallel(model)
        weights = torch.load(str(ckpt_path), map_location=device)
        if isinstance(weights, dict) and "state_dict" in weights and isinstance(weights["state_dict"], dict):
            weights = weights["state_dict"]
        if not isinstance(weights, dict):
            raise RuntimeError(f"Unsupported MODNet checkpoint format: {ckpt_path}")
        weights = _normalize_state_dict_keys(weights, want_module_prefix=True)
        model.load_state_dict(weights, strict=True)
        self.model = model.to(device=device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict_mask(
        self,
        image_bgr: np.ndarray,
        *,
        hint_mask_u8: np.ndarray | None = None,
        ref_size: int = 512,
        threshold: float = 0.5,
    ) -> np.ndarray:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = (x - 0.5) / 0.5
        x = x.unsqueeze(0)

        ref = max(int(ref_size), 32)
        if max(h, w) < ref or min(h, w) > ref:
            if w >= h:
                rh = ref
                rw = int(round((float(w) / float(h)) * ref))
            else:
                rw = ref
                rh = int(round((float(h) / float(w)) * ref))
        else:
            rh, rw = h, w
        rw = max(32, rw - (rw % 32))
        rh = max(32, rh - (rh % 32))
        x = F.interpolate(x, size=(rh, rw), mode="area")
        x = x.to(device=self.device, dtype=torch.float32)

        _, _, matte = self.model(x, True)
        matte = F.interpolate(matte, size=(h, w), mode="bilinear", align_corners=False)
        alpha = matte[0, 0].detach().cpu().numpy().clip(0.0, 1.0)
        mask = (alpha >= float(np.clip(float(threshold), 0.0, 1.0))).astype(np.uint8) * 255
        ref_mask = None
        if hint_mask_u8 is not None and int((hint_mask_u8 > 0).sum()) > 0:
            ref_mask = (hint_mask_u8 > 0).astype(np.uint8) * 255
            if ref_mask.shape[:2] != (h, w):
                ref_mask = cv2.resize(ref_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return _clean_mask(mask, ref_mask_u8=ref_mask)


def _get_runtime(ckpt_path: str | Path | None = None, device: str | torch.device | None = None) -> _MODNetRuntime:
    resolved_ckpt = resolve_modnet_ckpt_path(ckpt_path)
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    key = (str(resolved_ckpt), str(resolved_device))
    runtime = _MODNET_RUNTIME.get(key)
    if runtime is not None:
        return runtime
    with _MODNET_LOCK:
        runtime = _MODNET_RUNTIME.get(key)
        if runtime is None:
            runtime = _MODNetRuntime(resolved_ckpt, resolved_device)
            _MODNET_RUNTIME[key] = runtime
    return runtime


def estimate_person_mask_modnet(
    image_bgr: np.ndarray,
    *,
    hint_mask_u8: np.ndarray | None = None,
    ref_size: int = 512,
    threshold: float = 0.5,
    ckpt_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    runtime = _get_runtime(ckpt_path=ckpt_path, device=device)
    return runtime.predict_mask(
        image_bgr,
        hint_mask_u8=hint_mask_u8,
        ref_size=int(ref_size),
        threshold=float(threshold),
    )
