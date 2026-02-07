from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def read_image_rgb(path: str | Path) -> torch.Tensor:
    """Read an image and return float tensor in [0,1], shape (3,H,W)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def axis_angle_to_rotmat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (3,) axis-angle.
    Fully differentiable. Avoid torch.tensor([...]) with tensor entries (causes warnings).
    """
    theta = torch.norm(w) + 1e-8
    k = w / theta  # (3,)

    kx, ky, kz = k[0], k[1], k[2]
    zeros = torch.zeros((), device=w.device, dtype=w.dtype)

    # Skew-symmetric matrix of k
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky]),
            torch.stack([kz, zeros, -kx]),
            torch.stack([-ky, kx, zeros]),
        ],
        dim=0,
    )

    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    R = I + s * K + (1.0 - c) * (K @ K)
    return R


def train_gaussians_3d_learnable_camera(
    g: Gaussian3D,
    frames_dir: str | Path,
    frames_list: list[str],
    K_list: list[float],
    out_ckpt: str | Path,
    iters: int = 1000,
    lr_gauss: float = 1e-2,
    lr_cam: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # ---- New knobs (safe defaults) ----
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),  # match your white-ish dataset background
    cam_warmup_iters: int = 5000,  # freeze camera for first N iterations
    l1_weight: float = 0.25,       # add L1 to reduce blur
    use_alpha_mask: bool = True,   # suppress background learning
    alpha_threshold: float = 0.15, # mask threshold
    # ---- Camera mode / regularization ----
    camera_mode: str = "learnable",  # learnable | fixed
    cam_t_prior: tuple[float, float, float] = (0.0, 0.0, 2.5),
    cam_reg_weight: float = 0.0,
    # ---- Keypoint bbox mask ----
    keypoints_json: str | Path | None = None,
    kpt_conf_min: float = 0.2,
    kpt_box_pad: float = 0.15,
    use_kpt_bbox_mask: bool = True,
    # ---- Face emphasis (uses keypoints, no hard mask) ----
    face_weight: float = 1.0,
    face_conf_min: float = 0.3,
    face_box_pad: float = 0.2,
    # ---- Frame sampling / weighting ----
    frame_weights: list[float] | None = None,
    frame_loss_weights: list[float] | None = None,
    # ---- Dynamic binding to SMPL mesh sequence ----
    bind_mesh_dir: str | Path | None = None,
    bind_vertex_indices: torch.Tensor | None = None,
    bind_mesh_indices: list[int] | None = None,
    bind_cache_size: int = 8,
    bind_mesh_to_camera: bool = False,
    # ---- Background behavior for unbound gaussians ----
    bg_grad_scale: float = 1.0,
    bg_opacity_logit_max: float | None = None,
    bg_log_scale_min: float | None = None,
    bg_log_scale_max: float | None = None,
    # ---- Foreground (bound) constraints ----
    fg_opacity_logit_max: float | None = None,
    fg_log_scale_min: float | None = None,
    fg_log_scale_max: float | None = None,
):
    """
    Stage2: optimize 3D Gaussians + (optional) per-frame learnable camera.

    Key fixes vs the original:
    - Use white background compositing to match GT background (reduces colored speckles).
    - Freeze camera for warmup_iters, then enable camera learning (stabilizes convergence).
    - Add L1 term to reduce MSE-induced blur.
    - Random frame sampling (remove periodic bias).
    - Optional alpha-based masking to avoid fitting background.
    """

    frames_dir = Path(frames_dir)
    if len(frames_list) == 0:
        raise ValueError("frames_list is empty.")

    # Intrinsics
    K = torch.tensor(K_list, dtype=torch.float32, device=device).view(3, 3)

    # Load first frame for H,W
    gt0 = read_image_rgb(frames_dir / frames_list[0]).to(device)
    _, H, W = gt0.shape

    # Move gaussian to device and make trainable
    g = g.to(device)
    # If binding is enabled, we will override xyz per-frame from mesh; do NOT optimize xyz.
    binding_enabled = bind_mesh_dir is not None and bind_vertex_indices is not None
    if binding_enabled:
        g.xyz = g.xyz.detach()  # keep as buffer
    else:
        g.xyz = torch.nn.Parameter(g.xyz)
    g.log_scales = torch.nn.Parameter(g.log_scales)
    g.quat = torch.nn.Parameter(g.quat)
    g.opacity_logit = torch.nn.Parameter(g.opacity_logit)
    g.rgb_logit = torch.nn.Parameter(g.rgb_logit)

    # Camera per-frame (axis-angle + translation)
    nF = len(frames_list)
    cam_t_init = torch.tensor([cam_t_prior], device=device).float().view(1, 3)
    cam_w = torch.zeros((nF, 3), device=device)
    cam_t = cam_t_init.repeat(nF, 1)
    learn_cam = str(camera_mode).lower() != "fixed" and lr_cam > 0
    if learn_cam:
        cam_w = torch.nn.Parameter(cam_w)
        cam_t = torch.nn.Parameter(cam_t)

    # Optimizer (we will dynamically freeze/unfreeze camera by toggling requires_grad)
    gauss_params = [g.log_scales, g.quat, g.opacity_logit, g.rgb_logit]
    if not binding_enabled:
        gauss_params = [g.xyz] + gauss_params
    param_groups = [{"params": gauss_params, "lr": lr_gauss}]
    if learn_cam:
        param_groups.append({"params": [cam_w, cam_t], "lr": lr_cam})
    opt = torch.optim.Adam(param_groups)

    # Preload GT frames (ok for small clips; for long clips stream from disk)
    gts = [read_image_rgb(frames_dir / fn).to(device) for fn in frames_list]

    # Helper to freeze/unfreeze cam params
    def _set_cam_trainable(flag: bool):
        if learn_cam:
            cam_w.requires_grad_(flag)
            cam_t.requires_grad_(flag)

    # Warmup: freeze camera first
    _set_cam_trainable(False)

    # Optional: keypoint-based bbox mask
    kpt_boxes = None
    face_boxes = None
    if keypoints_json is not None:
        kp_path = Path(keypoints_json)
        if kp_path.exists():
            with kp_path.open("r", encoding="utf-8") as f:
                kpts = json.load(f)
            # Build per-frame bbox (x0,y0,x1,y1)
            kpt_boxes = [] if use_kpt_bbox_mask else None
            face_boxes = [] if float(face_weight) > 1.0 else None
            face_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # MediaPipe face/ears/mouth landmarks
            for fn in frames_list:
                arr = kpts.get(fn, None)
                if arr is None:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    continue
                xs = [p[0] for p in arr if p[2] >= kpt_conf_min]
                ys = [p[1] for p in arr if p[2] >= kpt_conf_min]
                if not xs or not ys:
                    if kpt_boxes is not None:
                        kpt_boxes.append(None)
                    if face_boxes is not None:
                        face_boxes.append(None)
                    continue
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                # pad bbox
                bw = (x1 - x0)
                bh = (y1 - y0)
                x0 = max(0.0, x0 - kpt_box_pad * bw)
                x1 = min(W - 1.0, x1 + kpt_box_pad * bw)
                y0 = max(0.0, y0 - kpt_box_pad * bh)
                y1 = min(H - 1.0, y1 + kpt_box_pad * bh)
                if kpt_boxes is not None:
                    kpt_boxes.append((x0, y0, x1, y1))

                if face_boxes is not None:
                    fx = []
                    fy = []
                    for fi in face_ids:
                        if fi < len(arr) and arr[fi][2] >= face_conf_min:
                            fx.append(arr[fi][0])
                            fy.append(arr[fi][1])
                    if not fx or not fy:
                        face_boxes.append(None)
                    else:
                        fx0, fx1 = min(fx), max(fx)
                        fy0, fy1 = min(fy), max(fy)
                        fbw = (fx1 - fx0)
                        fbh = (fy1 - fy0)
                        fx0 = max(0.0, fx0 - face_box_pad * fbw)
                        fx1 = min(W - 1.0, fx1 + face_box_pad * fbw)
                        fy0 = max(0.0, fy0 - face_box_pad * fbh)
                        fy1 = min(H - 1.0, fy1 + face_box_pad * fbh)
                        face_boxes.append((fx0, fy0, fx1, fy1))

    # Optional: small robustness—randomize frame order sampling
    # Optional: frame sampling weights (e.g., emphasize the last frame)
    frame_weights_t = None
    frame_loss_weights_t = None
    if frame_weights is not None:
        if len(frame_weights) != nF:
            raise ValueError("frame_weights length must match number of frames.")
        w = torch.tensor(frame_weights, dtype=torch.float32)
        if (w <= 0).any():
            raise ValueError("frame_weights must be > 0.")
        frame_weights_t = w / w.sum()
    if frame_loss_weights is not None:
        if len(frame_loss_weights) != nF:
            raise ValueError("frame_loss_weights length must match number of frames.")
        frame_loss_weights_t = torch.tensor(frame_loss_weights, dtype=torch.float32)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    # Binding helpers (lazy-load mesh vertices per frame)
    mesh_cache: dict[int, torch.Tensor] = {}
    mesh_cache_order: list[int] = []
    bind_idx_t = None
    if binding_enabled:
        bind_idx_t = bind_vertex_indices.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0
        if bind_mesh_indices is None:
            bind_mesh_indices = list(range(len(frames_list)))
        if len(bind_mesh_indices) != len(frames_list):
            raise ValueError("bind_mesh_indices must align with frames_list length.")
        bind_mesh_dir = Path(bind_mesh_dir)
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(f"bind_mesh_dir not found: {bind_mesh_dir}")

        def _load_mesh_verts(mesh_idx: int) -> torch.Tensor:
            if mesh_idx in mesh_cache:
                return mesh_cache[mesh_idx]
            obj_path = bind_mesh_dir / f"{mesh_idx:06d}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(f"Mesh frame not found: {obj_path}")
            verts_np = load_obj_vertices(obj_path)
            if bind_mesh_to_camera:
                # Inverse of stage1 _camera_to_blender: (x, y, z) = (x_b, -z_b, y_b)
                verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
            verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
            mesh_cache[mesh_idx] = verts_t
            mesh_cache_order.append(mesh_idx)
            if len(mesh_cache_order) > int(bind_cache_size):
                old = mesh_cache_order.pop(0)
                mesh_cache.pop(old, None)
            return verts_t

    for it in tqdm(range(iters), desc="Stage2 3DGS (learnable camera)"):
        opt.zero_grad(set_to_none=True)

        # Enable camera learning after warmup
        if learn_cam and it == cam_warmup_iters:
            _set_cam_trainable(True)

        # Sample a random frame index (better than it % nF)
        if frame_weights_t is None:
            fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        else:
            fi = int(torch.multinomial(frame_weights_t, 1, replacement=True, generator=rng).item())
        gt = gts[fi]  # (3,H,W)

        # Camera pose
        R = axis_angle_to_rotmat(cam_w[fi])
        t = cam_t[fi]

        # Update bound xyz for this frame (dynamic binding)
        if binding_enabled:
            mesh_idx = int(bind_mesh_indices[fi])
            verts_t = _load_mesh_verts(mesh_idx)
            if bind_mask.any():
                # Update only bound gaussians (bind_idx >= 0)
                g.xyz[bind_mask] = verts_t.index_select(0, bind_idx_t[bind_mask]).detach()

        # Render
        if use_alpha_mask:
            pred_rgb, pred_a = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=True)
            # pred_rgb: (3,H,W), pred_a: (1,H,W)
            # Build a mask to suppress background. Use detached alpha to avoid degenerate solutions.
            mask = (pred_a.detach() > alpha_threshold).float()  # (1,H,W)
        else:
            pred_rgb = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=False)
            mask = None

        # Loss: MSE + L1 (reduces blur)
        diff = pred_rgb - gt
        l2 = (diff * diff)
        l1 = diff.abs()

        # Optional keypoint bbox mask
        if use_kpt_bbox_mask and kpt_boxes is not None and kpt_boxes[fi] is not None:
            x0, y0, x1, y1 = kpt_boxes[fi]
            m2d = torch.zeros((1, H, W), device=device)
            m2d[:, int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1.0
            mask = m2d if mask is None else (mask * m2d)

        # Optional face emphasis (no hard mask, just higher weight)
        weight = None
        if face_boxes is not None and face_boxes[fi] is not None and float(face_weight) > 1.0:
            fx0, fy0, fx1, fy1 = face_boxes[fi]
            w2d = torch.ones((1, H, W), device=device)
            w2d[:, int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = float(face_weight)
            weight = w2d

        # Combine mask + weight into one weight map
        if mask is not None:
            weight = mask if weight is None else (weight * mask)
        if weight is not None:
            w = weight.expand_as(pred_rgb)
            l2 = (l2 * w).sum() / (w.sum() + 1e-6)
            l1 = (l1 * w).sum() / (w.sum() + 1e-6)
        else:
            l2 = l2.mean()
            l1 = l1.mean()

        loss = l2 + l1_weight * l1
        if frame_loss_weights_t is not None:
            loss = loss * float(frame_loss_weights_t[fi].item())
        if learn_cam and cam_reg_weight > 0:
            # Keep camera near prior (prevents drift)
            reg = (cam_w ** 2).mean() + ((cam_t - cam_t_init) ** 2).mean()
            loss = loss + float(cam_reg_weight) * reg

        loss.backward()
        # If we want background to be low-precision, down-scale its gradients.
        if binding_enabled and bg_grad_scale < 1.0:
            bg_mask = ~bind_mask
            with torch.no_grad():
                for p in (g.log_scales, g.quat, g.opacity_logit, g.rgb_logit):
                    if p.grad is None:
                        continue
                    p.grad[bg_mask] *= float(bg_grad_scale)
        opt.step()

        # Mild regularization: keep parameters in a reasonable range
        with torch.no_grad():
            g.log_scales.clamp_(min=-7.0, max=0.0)      # exp range ~ [0.0009, 1]
            g.opacity_logit.clamp_(min=-8.0, max=8.0)
            if binding_enabled:
                bg_mask = ~bind_mask
                # Foreground constraints (bound gaussians)
                if fg_opacity_logit_max is not None:
                    g.opacity_logit[bind_mask] = g.opacity_logit[bind_mask].clamp(
                        max=float(fg_opacity_logit_max)
                    )
                if fg_log_scale_min is not None or fg_log_scale_max is not None:
                    min_v = -7.0 if fg_log_scale_min is None else float(fg_log_scale_min)
                    max_v = 0.0 if fg_log_scale_max is None else float(fg_log_scale_max)
                    g.log_scales[bind_mask] = g.log_scales[bind_mask].clamp(
                        min=min_v, max=max_v
                    )
                if bg_opacity_logit_max is not None:
                    g.opacity_logit[bg_mask] = g.opacity_logit[bg_mask].clamp(
                        max=float(bg_opacity_logit_max)
                    )
                if bg_log_scale_min is not None or bg_log_scale_max is not None:
                    min_v = -7.0 if bg_log_scale_min is None else float(bg_log_scale_min)
                    max_v = 0.0 if bg_log_scale_max is None else float(bg_log_scale_max)
                    g.log_scales[bg_mask] = g.log_scales[bg_mask].clamp(
                        min=min_v, max=max_v
                    )
            # Optional: keep translations in a sane range to prevent explosion
            if learn_cam:
                cam_t[:, 2].clamp_(min=0.5, max=10.0)

    # Save checkpoint
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gaussians": g.state_dict(),
            "K": K.detach().cpu(),
            "cam_w": cam_w.detach().cpu() if torch.is_tensor(cam_w) else torch.tensor(cam_w),
            "cam_t": cam_t.detach().cpu() if torch.is_tensor(cam_t) else torch.tensor(cam_t),
            "frames": frames_list,
            "H": H,
            "W": W,
            "bg_rgb": bg_rgb,
            "cam_warmup_iters": cam_warmup_iters,
            "l1_weight": l1_weight,
            "use_alpha_mask": use_alpha_mask,
            "alpha_threshold": alpha_threshold,
            "camera_mode": camera_mode,
            "cam_t_prior": cam_t_prior,
            "cam_reg_weight": cam_reg_weight,
            "keypoints_json": str(keypoints_json) if keypoints_json is not None else None,
            "kpt_conf_min": kpt_conf_min,
            "kpt_box_pad": kpt_box_pad,
            "use_kpt_bbox_mask": use_kpt_bbox_mask,
            "face_weight": face_weight,
            "face_conf_min": face_conf_min,
            "face_box_pad": face_box_pad,
            "frame_weights": frame_weights,
            "frame_loss_weights": frame_loss_weights,
            "bind_mesh_dir": str(bind_mesh_dir) if binding_enabled else None,
            "bind_mesh_indices": bind_mesh_indices if binding_enabled else None,
            "bg_grad_scale": bg_grad_scale,
            "bg_opacity_logit_max": bg_opacity_logit_max,
            "bg_log_scale_min": bg_log_scale_min,
            "bg_log_scale_max": bg_log_scale_max,
            "fg_opacity_logit_max": fg_opacity_logit_max,
            "fg_log_scale_min": fg_log_scale_min,
            "fg_log_scale_max": fg_log_scale_max,
        },
        out_ckpt,
    )
