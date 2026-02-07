from __future__ import annotations

from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, resolve_path
from avatar_gs.utils.io import ensure_dir
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read image to (3,H,W) float in [0,1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle -> R."""
    theta = torch.norm(w) + 1e-8
    k = w / theta
    kx, ky, kz = k[0], k[1], k[2]
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    K = torch.stack(
        [
            torch.stack([z, -kz, ky]),
            torch.stack([kz, z, -kx]),
            torch.stack([-ky, kx, z]),
        ],
        dim=0,
    )
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return I + s * K + (1.0 - c) * (K @ K)


def save_rgb(path: str | Path, rgb_chw: torch.Tensor) -> None:
    """Save (3,H,W) RGB float [0,1] as png/jpg, preserving exact H/W."""
    img = (rgb_chw.detach().clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
    bgr = img[..., ::-1]  # BGR
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale camera intrinsics for image size change (or FOV scaling)."""
    if scale == 1.0:
        return K
    Ks = K.clone()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def _resize_chw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Resize (C,H,W) tensor to (C,H,W) with safe down/up-sampling."""
    if x.shape[1] == H and x.shape[2] == W:
        return x
    mode = "area" if (x.shape[1] > H or x.shape[2] > W) else "bilinear"
    if mode == "area":
        return F.interpolate(x.unsqueeze(0), size=(H, W), mode=mode).squeeze(0)
    return F.interpolate(x.unsqueeze(0), size=(H, W), mode=mode, align_corners=False).squeeze(0)


def _select_gaussians(g: Gaussian3D, mask: torch.Tensor) -> Gaussian3D:
    return Gaussian3D(
        xyz=g.xyz[mask],
        log_scales=g.log_scales[mask],
        quat=g.quat[mask],
        opacity_logit=g.opacity_logit[mask],
        rgb_logit=g.rgb_logit[mask],
        bind_idx=g.bind_idx[mask] if g.bind_idx is not None else None,
    )


def _load_keypoints_boxes(
    keypoints_json: Path,
    frames: list[str],
    H: int,
    W: int,
    conf_min: float,
    pad: float,
) -> list[tuple[float, float, float, float] | None]:
    if not keypoints_json.exists():
        raise FileNotFoundError(f"keypoints_json not found: {keypoints_json}")
    import json as _json

    with keypoints_json.open("r", encoding="utf-8") as f:
        kpts = _json.load(f)
    boxes = []
    for fn in frames:
        arr = kpts.get(fn, None)
        if arr is None:
            boxes.append(None)
            continue
        xs = [p[0] for p in arr if p[2] >= conf_min]
        ys = [p[1] for p in arr if p[2] >= conf_min]
        if not xs or not ys:
            boxes.append(None)
            continue
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = (x1 - x0)
        bh = (y1 - y0)
        x0 = max(0.0, x0 - pad * bw)
        x1 = min(W - 1.0, x1 + pad * bw)
        y0 = max(0.0, y0 - pad * bh)
        y1 = min(H - 1.0, y1 + pad * bh)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _resolve_stage2_ckpt(stage2_dir: Path) -> Path:
    """
    Find a checkpoint file inside stage2_dir.
    Tries common filenames first, then falls back to any .pt found.
    """
    common = [
        stage2_dir / "gaussians.pt",
        stage2_dir / "gaussians3d.pt",
        stage2_dir / "ckpt.pt",
    ]
    for p in common:
        if p.exists():
            return p

    pts = sorted(stage2_dir.glob("*.pt"))
    if pts:
        return pts[-1]

    ckpt_dir = stage2_dir / "checkpoints"
    if ckpt_dir.exists():
        pts2 = sorted(ckpt_dir.glob("*.pt"))
        if pts2:
            return pts2[-1]

    raise FileNotFoundError(f"No checkpoint (*.pt) found under: {stage2_dir}")


def main():
    parser = parse_args_base("Render images from Stage2 gaussian checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("render", str(log_dir / "render.log"))

    render_cfg = cfg["render"]
    stage2_dir = project_root / Path(render_cfg.get("stage2_dir", "outputs/stage2"))
    out_dir = ensure_dir(project_root / Path(render_cfg.get("out_dir", "outputs/renders")))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoint path: optional explicit, otherwise auto-resolve
    ckpt_path = render_cfg.get("ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).strip():
        ckpt_path = Path(resolve_path(str(ckpt_path)))
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
    else:
        ckpt_path = _resolve_stage2_ckpt(stage2_dir)

    logger.info(f"Stage2 dir: {stage2_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Out dir: {out_dir}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Load gaussians
    #g = Gaussian3D()
    #g.load_state_dict(ckpt["gaussians"])
    #g = g.to(device)

    # Load gaussians (Gaussian3D requires tensors in __init__)
    gsd = ckpt["gaussians"]  # this is g.state_dict() saved in stage2

    required = ["xyz", "log_scales", "quat", "opacity_logit", "rgb_logit"]
    missing = [k for k in required if k not in gsd]
    if missing:
      raise KeyError(f"Checkpoint gaussians missing keys: {missing}. Available keys: {list(gsd.keys())}")

    g = Gaussian3D(
      xyz=gsd["xyz"].to(device),
      log_scales=gsd["log_scales"].to(device),
      quat=gsd["quat"].to(device),
      opacity_logit=gsd["opacity_logit"].to(device),
      rgb_logit=gsd["rgb_logit"].to(device),
      bind_idx=gsd.get("bind_idx", None),
    )

    # Intrinsics
    K = torch.tensor(ckpt["K"], dtype=torch.float32, device=device).view(3, 3)

    # Cameras and frames
    frames = ckpt.get("frames", None)
    if frames is None or len(frames) == 0:
        raise KeyError("Checkpoint missing 'frames'. Cannot render without frame list.")

    cam_w = ckpt.get("cam_w", None)
    cam_t = ckpt.get("cam_t", None)
    cam_w = torch.tensor(cam_w, dtype=torch.float32, device=device) if cam_w is not None else None
    cam_t = torch.tensor(cam_t, dtype=torch.float32, device=device) if cam_t is not None else None

    # frames_dir: optional in config, otherwise default to data/frames/input
    frames_dir = render_cfg.get("frames_dir", None)
    if frames_dir is not None and str(frames_dir).strip():
        frames_dir = Path(resolve_path(str(frames_dir)))
        if not frames_dir.is_absolute():
            frames_dir = project_root / frames_dir
    else:
        frames_dir = project_root / Path("data/frames/input")

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    # How many frames to render
    max_frames = int(render_cfg.get("max_frames", -1))
    to_render = len(frames) if max_frames <= 0 else min(len(frames), max_frames)

    # Background (default white)
    bg = render_cfg.get("bg", [1.0, 1.0, 1.0])
    if isinstance(bg, (list, tuple)) and len(bg) == 3:
        bg = (float(bg[0]), float(bg[1]), float(bg[2]))
    else:
        bg = (1.0, 1.0, 1.0)

    logger.info(f"Frames dir: {frames_dir}")
    logger.info(f"Will render: {to_render} frames; bg={bg}")

    fg_scale = float(render_cfg.get("fg_scale", 1.0))
    bg_scale = float(render_cfg.get("bg_scale", 1.0))
    k_scale = float(render_cfg.get("k_scale", 1.0))
    split_by_bind = bool(render_cfg.get("split_by_bind", False))
    if fg_scale <= 0.0 or bg_scale <= 0.0:
        raise ValueError(f"fg_scale/bg_scale must be > 0. Got fg_scale={fg_scale}, bg_scale={bg_scale}")

    if fg_scale != 1.0 or bg_scale != 1.0:
        logger.info(f"Render scales: fg_scale={fg_scale}, bg_scale={bg_scale}")
    if k_scale != 1.0:
        logger.info(f"Intrinsic scale (FOV): k_scale={k_scale}")

    # Optional keypoint bbox mask for foreground/background split
    mask_mode = str(render_cfg.get("mask_mode", "alpha")).lower()
    kpt_boxes = None
    kp_json = render_cfg.get("keypoints_json", None)
    if kp_json is not None:
        kp_json = Path(resolve_path(str(kp_json)))
        if not kp_json.is_absolute():
            kp_json = project_root / kp_json

    # Optional: dynamic binding to SMPL mesh sequence
    bind_mesh_dir = render_cfg.get("bind_mesh_sequence_dir", None)
    bind_seq_npz = render_cfg.get("bind_sequence_npz", None)
    bind_mesh_indices = ckpt.get("bind_mesh_indices", None)
    binding_enabled = bind_mesh_dir is not None and g.bind_idx is not None
    bind_mesh_to_camera = bool(render_cfg.get("bind_mesh_to_camera", False))

    mesh_cache: dict[int, torch.Tensor] = {}
    mesh_cache_order: list[int] = []
    bind_idx_t = None
    if binding_enabled:
        bind_mesh_dir = Path(resolve_path(str(bind_mesh_dir)))
        if not bind_mesh_dir.is_absolute():
            bind_mesh_dir = project_root / bind_mesh_dir
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(f"bind_mesh_sequence_dir not found: {bind_mesh_dir}")

        if bind_mesh_indices is None:
            # try to map frame names -> mesh indices using smpl_params_sequence.npz
            if bind_seq_npz is None:
                bind_seq_npz = project_root / Path("outputs/stage1/smpl_params_sequence.npz")
            else:
                bind_seq_npz = Path(resolve_path(str(bind_seq_npz)))
                if not bind_seq_npz.is_absolute():
                    bind_seq_npz = project_root / bind_seq_npz
            if bind_seq_npz.exists():
                import numpy as _np
                seq = _np.load(bind_seq_npz, allow_pickle=True)
                seq_names = [str(x) for x in seq["frame_names"].tolist()]
                name_to_idx = {n: i for i, n in enumerate(seq_names)}
                bind_mesh_indices = []
                missing = []
                for fn in frames:
                    if fn in name_to_idx:
                        bind_mesh_indices.append(int(name_to_idx[fn]))
                    else:
                        missing.append(fn)
                if missing and len(seq_names) == len(frames):
                    bind_mesh_indices = list(range(len(frames)))
                elif missing:
                    raise RuntimeError(f"Binding failed: {len(missing)} frames not in {bind_seq_npz}")
            else:
                bind_mesh_indices = list(range(len(frames)))

        bind_idx_t = g.bind_idx.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0

        def _load_mesh_verts(mesh_idx: int) -> torch.Tensor:
            if mesh_idx in mesh_cache:
                return mesh_cache[mesh_idx]
            obj_path = bind_mesh_dir / f"{mesh_idx:06d}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(f"Mesh frame not found: {obj_path}")
            verts_np = load_obj_vertices(obj_path)
            if bind_mesh_to_camera:
                # Inverse of stage1 _camera_to_blender: (x, y, z) = (x_b, -z_b, y_b)
                import numpy as _np
                verts_np = _np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(_np.float32)
            verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
            mesh_cache[mesh_idx] = verts_t
            mesh_cache_order.append(mesh_idx)
            if len(mesh_cache_order) > 8:
                old = mesh_cache_order.pop(0)
                mesh_cache.pop(old, None)
            return verts_t

    for i in range(to_render):
        frame_path = frames_dir / frames[i]
        gt = read_image_rgb(frame_path).to(device)  # (3,H,W)
        base_h, base_w = int(gt.shape[1]), int(gt.shape[2])
        use_gt_bg = bool(render_cfg.get("use_gt_background", False))
        bg_gt_chw = gt if use_gt_bg else None
        bg_gt_hwc = gt.permute(1, 2, 0).contiguous() if use_gt_bg else None
        if mask_mode == "kpt" and kpt_boxes is None and kp_json is not None:
            kpt_boxes = _load_keypoints_boxes(
                keypoints_json=kp_json,
                frames=frames,
                H=base_h,
                W=base_w,
                conf_min=float(render_cfg.get("kpt_conf_min", 0.2)),
                pad=float(render_cfg.get("kpt_box_pad", 0.15)),
            )

        # Camera for this frame
        if cam_w is None or cam_t is None:
            R = torch.eye(3, device=device, dtype=torch.float32)
            t = torch.tensor([0.0, 0.0, 2.5], device=device, dtype=torch.float32)
        else:
            R = axis_angle_to_rotmat(cam_w[i])
            t = cam_t[i]

        # Update bound xyz for this frame (dynamic binding)
        if binding_enabled:
            mesh_idx = int(bind_mesh_indices[i])
            verts_t = _load_mesh_verts(mesh_idx)
            if bind_mask.any():
                g.xyz[bind_mask] = verts_t.index_select(0, bind_idx_t[bind_mask]).detach()

        # Optional: split gaussians by bind_idx
        if split_by_bind and g.bind_idx is not None:
            fg_mask = (g.bind_idx >= 0)
            bg_mask = ~fg_mask
            g_fg = _select_gaussians(g, fg_mask)
            g_bg = _select_gaussians(g, bg_mask) if bg_mask.any() else None
        else:
            g_fg = g
            g_bg = None

        # Scale intrinsics for wider/narrower FOV if requested
        K_use = _scale_intrinsics(K, k_scale)

        if fg_scale == 1.0 and bg_scale == 1.0:
            # IMPORTANT: infer output size from gt (do NOT hardcode H/W)
            pred = render_gaussians_3d(
                g=g_fg,
                K=K_use,
                R=R,
                t=t,
                H=None,
                W=None,
                bg=bg_gt_hwc if use_gt_bg else bg,
                ref_image=gt,  # << output size == gt size
            )
        else:
            # Foreground: higher scale, keep alpha for compositing
            fg_h = max(1, int(round(base_h * fg_scale)))
            fg_w = max(1, int(round(base_w * fg_scale)))
            K_fg = _scale_intrinsics(K_use, fg_scale)
            fg_rgb, fg_alpha = render_gaussians_3d(
                g=g_fg,
                K=K_fg,
                R=R,
                t=t,
                H=fg_h,
                W=fg_w,
                bg=None,
                return_alpha=True,
            )
            fg_rgb = _resize_chw(fg_rgb, base_h, base_w)
            fg_alpha = _resize_chw(fg_alpha, base_h, base_w).clamp(0.0, 1.0)

            # Background: either render low-res gaussians, or use GT image
            if use_gt_bg and bg_gt_chw is not None:
                bg_rgb = bg_gt_chw
            else:
                bg_h = max(1, int(round(base_h * bg_scale)))
                bg_w = max(1, int(round(base_w * bg_scale)))
                K_bg = _scale_intrinsics(K_use, bg_scale)
                bg_rgb = render_gaussians_3d(
                    g=g_bg if g_bg is not None else g_fg,
                    K=K_bg,
                    R=R,
                    t=t,
                    H=bg_h,
                    W=bg_w,
                    bg=bg,
                )
                bg_rgb = _resize_chw(bg_rgb, base_h, base_w)

            if mask_mode == "kpt" and kpt_boxes is not None and kpt_boxes[i] is not None:
                x0, y0, x1, y1 = kpt_boxes[i]
                m = torch.zeros((1, base_h, base_w), device=device)
                m[:, int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1.0
                pred = fg_rgb * m + bg_rgb * (1.0 - m)
            else:
                pred = fg_rgb * fg_alpha + bg_rgb * (1.0 - fg_alpha)

        out_path = out_dir / f"{i:06d}.png"
        save_rgb(out_path, pred)

        if (i + 1) % 50 == 0:
            logger.info(f"Rendered {i+1}/{to_render} ...")

    logger.info("Done.")


if __name__ == "__main__":
    main()
