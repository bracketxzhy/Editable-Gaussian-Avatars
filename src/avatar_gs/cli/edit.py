from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import cv2

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def _pick(d: Dict[str, Any], keys: Tuple[str, ...], required: bool = True):
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"Missing keys {keys} in checkpoint. Available keys: {list(d.keys())}")
    return None


def _axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues: w (...,3) -> R (...,3,3)
    """
    w = w.to(dtype=torch.float32)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz,  ky], dim=-1),
        torch.stack([kz,  zero, -kx], dim=-1),
        torch.stack([-ky, kx,  zero], dim=-1),
    ], dim=-2)  # (...,3,3)

    I = torch.eye(3, device=w.device, dtype=torch.float32).expand(K.shape)
    ct = torch.cos(theta)[..., None]
    st = torch.sin(theta)[..., None]
    R = I + st * K + (1.0 - ct) * (K @ K)
    return R


def _read_image_rgb_float(path: Path, H: int, W: int) -> torch.Tensor:
    """
    Read image -> RGB float32 in [0,1], shape (H,W,3).
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img_rgb).float() / 255.0
    return img  # (H,W,3)


def _apply_simple_edit(g: Gaussian3D, cfg: Dict[str, Any]) -> Gaussian3D:
    """
    “编辑”示例（基于 stage2 高斯）：
      - global_translate: [dx,dy,dz]
      - global_scale: s
      - rgb_gain: [gr,gg,gb]  (在 rgb 空间乘法，尽量轻微)
      - opacity_gain: s (在 alpha 空间乘法，尽量轻微)
    这些编辑都是可逆/可控的，不会破坏曝光匹配。
    """
    with torch.no_grad():
        # global scale/translate on xyz
        s = float(cfg.get("global_scale", 1.0))
        t = cfg.get("global_translate", [0.0, 0.0, 0.0])
        t = torch.as_tensor(t, device=g.xyz.device, dtype=g.xyz.dtype).view(1, 3)

        xyz = g.xyz * s + t

        # mild rgb gain in rgb-space (not logit) then map back to logits safely
        rgb_gain = cfg.get("rgb_gain", [1.0, 1.0, 1.0])
        rgb_gain = torch.as_tensor(rgb_gain, device=g.xyz.device, dtype=g.xyz.dtype).view(1, 3)

        rgb = torch.sigmoid(g.rgb_logit) * rgb_gain
        rgb = rgb.clamp(1e-4, 1.0 - 1e-4)
        rgb_logit = torch.log(rgb / (1.0 - rgb))

        # mild opacity gain in alpha-space then back to logits
        opacity_gain = float(cfg.get("opacity_gain", 1.0))
        a = torch.sigmoid(g.opacity_logit) * opacity_gain
        a = a.clamp(1e-4, 1.0 - 1e-4)
        opacity_logit = torch.log(a / (1.0 - a))

    return Gaussian3D(
        xyz=xyz,
        log_scales=g.log_scales,
        quat=g.quat,
        opacity_logit=opacity_logit,
        rgb_logit=rgb_logit,
        bind_idx=g.bind_idx,
        bind_offset=g.bind_offset,
        bind_face_idx=g.bind_face_idx,
        bind_bary=g.bind_bary,
        bind_hybrid_w=g.bind_hybrid_w,
    )


def main():
    parser = parse_args_base("Edit stage2 gaussians and render with original-like exposure")
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)

    root = Path(cfg.get("project_root", ".")).resolve()
    logger = setup_logger("edit", None)

    # ---- paths ----
    stage2_dir = root / cfg.get("stage2_dir", "outputs/stage2")
    ckpt_path = stage2_dir / cfg.get("stage2_ckpt", "gaussians3d.pt")
    out_dir = ensure_dir(root / cfg.get("out_dir", "outputs/edits"))

    # ---- render size ----
    H = int(cfg.get("H", cfg.get("height", 480)))
    W = int(cfg.get("W", cfg.get("width", 640)))

    # ---- load checkpoint ----
    ckpt: Dict[str, Any] = torch.load(str(ckpt_path), map_location="cpu")

    gauss_sd = _pick(ckpt, ("gaussians", "gaussian", "g"), required=True)
    K_ckpt = _pick(ckpt, ("K", "intrinsics"), required=True)
    cam_w = _pick(ckpt, ("cam_w", "camera_w", "w"), required=True)
    cam_t = _pick(ckpt, ("cam_t", "camera_t", "t"), required=True)
    frames = _pick(ckpt, ("frames", "frame_names"), required=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gauss = Gaussian3D.load_state_dict(gauss_sd).to(str(device))
    K = K_ckpt.to(device=device, dtype=torch.float32)

    # ---- select frame ----
    frame_idx = int(cfg.get("frame_idx", 0))
    if torch.is_tensor(cam_w):
        w_i = cam_w[frame_idx].to(device=device)
    else:
        w_i = torch.as_tensor(cam_w[frame_idx], device=device, dtype=torch.float32)

    if torch.is_tensor(cam_t):
        t_i = cam_t[frame_idx].to(device=device)
    else:
        t_i = torch.as_tensor(cam_t[frame_idx], device=device, dtype=torch.float32)

    # cam_w can be axis-angle (3,) or rotation matrix (3,3)
    if w_i.ndim == 1 and w_i.numel() == 3:
        R = _axis_angle_to_matrix(w_i).to(device=device, dtype=torch.float32)
    elif w_i.ndim == 2 and tuple(w_i.shape) == (3, 3):
        R = w_i.to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported cam_w shape: {tuple(w_i.shape)} (expect (3,) axis-angle or (3,3) matrix)")

    t = t_i.to(device=device, dtype=torch.float32).view(3)

    # ---- optional: load GT background frame for exposure/color matching ----
    use_gt_bg = bool(cfg.get("use_gt_background", True))
    bg = (0.0, 0.0, 0.0)

    frames_dir = cfg.get("frames_dir", None)
    if use_gt_bg and frames_dir is not None:
        frames_dir = (root / frames_dir).resolve()

        # try to pick filename from ckpt frames list if available; else fall back to pattern
        if isinstance(frames, (list, tuple)) and len(frames) > frame_idx:
            fname = frames[frame_idx]
            img_path = frames_dir / fname
            if not img_path.exists():
                # try common extensions if fname has no suffix
                if img_path.suffix == "":
                    for ext in (".png", ".jpg", ".jpeg"):
                        p = img_path.with_suffix(ext)
                        if p.exists():
                            img_path = p
                            break
        else:
            # fallback: 000000.png
            img_path = frames_dir / f"{frame_idx:06d}.png"
            if not img_path.exists():
                img_path = frames_dir / f"{frame_idx:06d}.jpg"

        if img_path.exists():
            bg = _read_image_rgb_float(img_path, H=H, W=W).to(device=device, dtype=torch.float32)
            logger.info(f"Using GT background: {img_path}")
        else:
            logger.warning(f"GT background requested but not found. frames_dir={frames_dir}, idx={frame_idx}. Using black bg.")

    # ---- apply edit on stage2 gaussians (in-place style edit) ----
    edit_cfg = cfg.get("edit", {})
    gauss_edit = _apply_simple_edit(gauss, edit_cfg)

    # ---- render (background is GT frame -> exposure naturally close to original) ----
    rgb = render_gaussians_3d(
        gauss_edit,
        K=K,
        R=R,
        t=t,
        H=H,
        W=W,
        bg=bg,
    ).detach().clamp(0, 1)  # (3,H,W)

    # save
    img = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")  # HWC RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out_path = out_dir / f"edit_{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), img)
    logger.info(f"Saved edited render: {out_path}")


if __name__ == "__main__":
    main()
