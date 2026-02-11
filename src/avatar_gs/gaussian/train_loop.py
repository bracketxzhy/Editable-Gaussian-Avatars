from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices, load_obj_mesh
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


def _project_points_numpy(
    xyz: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points with camera convention X_cam = R * X + t.
    Returns (u, v, valid_z_mask).
    """
    Xc = xyz @ R.T + t.reshape(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-6
    u = np.zeros((xyz.shape[0],), dtype=np.float32)
    v = np.zeros((xyz.shape[0],), dtype=np.float32)
    if np.any(valid):
        Xcv = Xc[valid]
        zv = z[valid]
        u[valid] = K[0, 0] * (Xcv[:, 0] / zv) + K[0, 2]
        v[valid] = K[1, 1] * (Xcv[:, 1] / zv) + K[1, 2]
    return u, v, valid


def _rasterize_mesh_mask(
    verts: np.ndarray,
    faces: np.ndarray | None,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    dilate_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    """
    Rasterize mesh silhouette to a binary mask (H, W) in {0,1}.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if faces is None or len(faces) == 0 or verts.shape[0] == 0:
        return mask

    u, v, valid = _project_points_numpy(verts, K, R, t)
    if not np.any(valid):
        return mask

    uvi = np.stack([u, v], axis=1).astype(np.float32)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if i0 < 0 or i1 < 0 or i2 < 0:
            continue
        if i0 >= len(valid) or i1 >= len(valid) or i2 >= len(valid):
            continue
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        pts = uvi[[i0, i1, i2]]
        if (
            (pts[:, 0] < -2).all()
            or (pts[:, 0] > W + 1).all()
            or (pts[:, 1] < -2).all()
            or (pts[:, 1] > H + 1).all()
        ):
            continue
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(mask, poly, 1)

    if int(dilate_px) > 0:
        k = int(dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if int(erode_px) > 0:
        k = int(erode_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def _erode_binary_mask_torch(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    """
    Erode a binary-like mask tensor (1,H,W) using max-pool based morphology on GPU.
    """
    if int(erode_px) <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"mask must be (1,H,W), got {mask.shape}")
    k = int(erode_px) * 2 + 1
    inv = (1.0 - mask).clamp(0.0, 1.0).unsqueeze(0)  # (1,1,H,W)
    dil_inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=int(erode_px))
    out = 1.0 - dil_inv.squeeze(0)
    return out.clamp(0.0, 1.0)


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
    cam_t_prior: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
    # ---- Temporal reference for current frame ----
    temporal_ref_window: int = 0,
    temporal_ref_decay: float = 0.85,
    temporal_ref_anchor: str = "sample",  # sample | last
    # ---- Dynamic binding to SMPL mesh sequence ----
    bind_mesh_dir: str | Path | None = None,
    bind_vertex_indices: torch.Tensor | None = None,
    bind_mesh_indices: list[int] | None = None,
    bind_cache_size: int = 8,
    bind_seq_npz: str | Path | None = None,
    bind_vertices_npz: str | Path | None = None,
    bind_mesh_to_camera: bool = False,
    bind_inverse_export_upright: bool = True,
    bind_mesh_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    # ---- SMPL silhouette supervision for bound human gaussians ----
    use_smpl_silhouette_mask: bool = False,
    smpl_mask_dilate_px: int = 0,
    smpl_mask_erode_px: int = 0,
    smpl_color_mask_erode_px: int = 0,
    smpl_alpha_leak_weight: float = 0.0,
    smpl_mask_override_alpha: bool = True,
    smpl_mask_precompute: bool = True,
    smpl_mask_precompute_max_frames: int = 64,
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
    - Optional SMPL silhouette masking/leak penalty to keep bound gaussians on the body region.
    - For fixed camera, optional precompute of silhouette masks to reduce per-iteration CPU work.
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
    if bool(use_smpl_silhouette_mask) and not binding_enabled:
        raise ValueError("use_smpl_silhouette_mask=true requires bind_mesh_dir + bind_vertex_indices.")
    if int(smpl_mask_dilate_px) < 0 or int(smpl_mask_erode_px) < 0 or int(smpl_color_mask_erode_px) < 0:
        raise ValueError(
            "smpl_mask_dilate_px/smpl_mask_erode_px/smpl_color_mask_erode_px must be >= 0. "
            f"Got {smpl_mask_dilate_px}/{smpl_mask_erode_px}/{smpl_color_mask_erode_px}"
        )
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
    ref_window = int(temporal_ref_window)
    if ref_window < 0:
        raise ValueError(f"temporal_ref_window must be >= 0. Got {temporal_ref_window}")
    ref_decay = float(temporal_ref_decay)
    if ref_window > 0 and (ref_decay <= 0.0 or ref_decay > 1.0):
        raise ValueError(
            f"temporal_ref_decay must be in (0,1] when temporal_ref_window>0. Got {temporal_ref_decay}"
        )
    ref_anchor = str(temporal_ref_anchor).lower()
    if ref_anchor not in ("sample", "last"):
        raise ValueError(f"temporal_ref_anchor must be sample|last. Got {temporal_ref_anchor}")

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
    smpl_mask_cache: dict[int, torch.Tensor] = {}
    bind_idx_t = None
    bind_vertex_offset_t = None
    bind_faces_np = None
    K_np = K.detach().cpu().numpy().astype(np.float32)
    if binding_enabled:
        bind_idx_t = bind_vertex_indices.to(device=device, dtype=torch.long)
        bind_mask = bind_idx_t >= 0
        if g.bind_offset is not None:
            bind_vertex_offset_t = g.bind_offset.to(device=device, dtype=torch.float32)
        bind_mesh_offset_t = torch.tensor(bind_mesh_offset, dtype=torch.float32, device=device).view(1, 3)
        exp_upright_rot = np.eye(3, dtype=np.float32)
        exp_center_xy = np.zeros((2,), dtype=np.float32)
        exp_floor_z = 0.0
        exp_has_center_xy = False
        exp_has_floor = False
        if bind_mesh_indices is None:
            bind_mesh_indices = list(range(len(frames_list)))
        if len(bind_mesh_indices) != len(frames_list):
            raise ValueError("bind_mesh_indices must align with frames_list length.")
        verts_seq_np = None
        verts_seq_ids_set: set[int] = set()
        verts_seq_fallback_idx = 0
        if bind_vertices_npz is not None:
            vseq_path = Path(bind_vertices_npz)
            if vseq_path.exists():
                vseq = np.load(vseq_path, allow_pickle=True)
                if "verts_cam" in vseq.files:
                    verts_seq_np = np.asarray(vseq["verts_cam"], dtype=np.float32)
                    if verts_seq_np.ndim != 3 or verts_seq_np.shape[-1] != 3:
                        raise RuntimeError(
                            f"Bad verts_cam shape in {vseq_path}: {verts_seq_np.shape}"
                        )
                    verts_seq_ids_set = set(range(int(verts_seq_np.shape[0])))
                    verts_seq_fallback_idx = int(max(int(verts_seq_np.shape[0]) - 1, 0))
                    print(f"Binding uses dynamic vertices from {vseq_path}")
                else:
                    print(f"Warning: {vseq_path} missing verts_cam; fallback to OBJ sequence binding.")
            else:
                print(f"Warning: bind_vertices_npz not found: {vseq_path}. Fallback to OBJ sequence binding.")
        bind_mesh_dir = Path(bind_mesh_dir)
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(f"bind_mesh_dir not found: {bind_mesh_dir}")
        mesh_ids = []
        for p in bind_mesh_dir.glob("*.obj"):
            stem = p.stem
            if stem.isdigit():
                mesh_ids.append(int(stem))
        if len(mesh_ids) == 0:
            raise FileNotFoundError(f"No numeric OBJ meshes found in bind_mesh_dir: {bind_mesh_dir}")
        mesh_ids_set = set(mesh_ids)
        fallback_mesh_idx = int(max(mesh_ids))
        if verts_seq_np is not None:
            missing_seq_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in verts_seq_ids_set]
            if len(missing_seq_ids) > 0:
                print(
                    "Warning: some bind indices are missing in verts sequence; "
                    f"fallback to frame {verts_seq_fallback_idx} for {len(missing_seq_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in verts_seq_ids_set else verts_seq_fallback_idx
                    for mi in bind_mesh_indices
                ]
        else:
            missing_mesh_ids = [int(mi) for mi in bind_mesh_indices if int(mi) not in mesh_ids_set]
            if len(missing_mesh_ids) > 0:
                print(
                    "Warning: some bind mesh indices are missing; "
                    f"fallback to mesh {fallback_mesh_idx:06d}.obj for {len(missing_mesh_ids)} frames."
                )
                bind_mesh_indices = [
                    int(mi) if int(mi) in mesh_ids_set else fallback_mesh_idx
                    for mi in bind_mesh_indices
                ]
        if bool(use_smpl_silhouette_mask):
            if len(bind_mesh_indices) == 0:
                raise RuntimeError("use_smpl_silhouette_mask=true but bind_mesh_indices is empty.")
            first_mesh_idx = int(bind_mesh_indices[0]) if int(bind_mesh_indices[0]) in mesh_ids_set else fallback_mesh_idx
            first_obj = bind_mesh_dir / f"{first_mesh_idx:06d}.obj"
            if not first_obj.exists():
                raise FileNotFoundError(f"Mesh frame not found for silhouette mask: {first_obj}")
            _, bind_faces_np = load_obj_mesh(first_obj)
            if bind_faces_np is None or len(bind_faces_np) == 0:
                raise RuntimeError(
                    "use_smpl_silhouette_mask=true but mesh has no faces; "
                    "cannot build silhouette."
                )

        if bind_mesh_to_camera and bind_inverse_export_upright and bind_seq_npz is not None:
            seq_path = Path(bind_seq_npz)
            if seq_path.exists():
                seq = np.load(seq_path, allow_pickle=True)
                if "export_upright_rot" in seq.files:
                    exp_upright_rot = np.asarray(seq["export_upright_rot"], dtype=np.float32).reshape(3, 3)
                if "export_center_xy" in seq.files:
                    exp_has_center_xy = bool(int(np.asarray(seq["export_center_xy"]).reshape(-1)[0]))
                if "export_center_xy_value" in seq.files:
                    exp_center_xy = np.asarray(seq["export_center_xy_value"], dtype=np.float32).reshape(2,)
                if "export_on_floor" in seq.files:
                    exp_has_floor = bool(int(np.asarray(seq["export_on_floor"]).reshape(-1)[0]))
                if "export_floor_z" in seq.files:
                    exp_floor_z = float(np.asarray(seq["export_floor_z"], dtype=np.float32).reshape(-1)[0])

        def _load_mesh_verts(mesh_idx: int) -> torch.Tensor:
            if verts_seq_np is not None:
                mesh_idx_use = int(mesh_idx)
                if mesh_idx_use not in verts_seq_ids_set:
                    mesh_idx_use = verts_seq_fallback_idx
                if mesh_idx_use in mesh_cache:
                    return mesh_cache[mesh_idx_use]
                verts_np = verts_seq_np[mesh_idx_use]
                verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
                verts_t = verts_t + bind_mesh_offset_t
                mesh_cache[mesh_idx_use] = verts_t
                mesh_cache_order.append(mesh_idx_use)
                if len(mesh_cache_order) > int(bind_cache_size):
                    old = mesh_cache_order.pop(0)
                    mesh_cache.pop(old, None)
                return verts_t

            mesh_idx_use = int(mesh_idx)
            if mesh_idx_use not in mesh_ids_set:
                mesh_idx_use = fallback_mesh_idx
            if mesh_idx_use in mesh_cache:
                return mesh_cache[mesh_idx_use]
            obj_path = bind_mesh_dir / f"{mesh_idx_use:06d}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(f"Mesh frame not found: {obj_path}")
            verts_np = load_obj_vertices(obj_path)
            if bind_mesh_to_camera:
                if bind_inverse_export_upright:
                    if exp_has_center_xy:
                        verts_np[:, 0] += float(exp_center_xy[0])
                        verts_np[:, 1] += float(exp_center_xy[1])
                    if exp_has_floor:
                        verts_np[:, 2] += float(exp_floor_z)
                    # Inverse of stage1 upright rotation: v_blender_raw = v_export @ U
                    verts_np = (verts_np @ exp_upright_rot.astype(np.float32))
                # Inverse of stage1 _camera_to_blender: (x, y, z) = (x_b, -z_b, y_b)
                verts_np = np.stack([verts_np[:, 0], -verts_np[:, 2], verts_np[:, 1]], axis=-1).astype(np.float32)
            verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
            verts_t = verts_t + bind_mesh_offset_t
            mesh_cache[mesh_idx_use] = verts_t
            mesh_cache_order.append(mesh_idx_use)
            if len(mesh_cache_order) > int(bind_cache_size):
                old = mesh_cache_order.pop(0)
                mesh_cache.pop(old, None)
            return verts_t

        if (
            bool(use_smpl_silhouette_mask)
            and bind_faces_np is not None
            and (not learn_cam)
            and bool(smpl_mask_precompute)
            and int(smpl_mask_precompute_max_frames) > 0
            and len(bind_mesh_indices) <= int(smpl_mask_precompute_max_frames)
        ):
            cam_r0 = axis_angle_to_rotmat(cam_w[0]).detach().cpu().numpy().astype(np.float32)
            cam_t0 = cam_t[0].detach().cpu().numpy().astype(np.float32)
            unique_mesh = sorted(set(int(mi) for mi in bind_mesh_indices))
            for mesh_idx_pc in unique_mesh:
                verts_pc = _load_mesh_verts(mesh_idx_pc)
                mask_np = _rasterize_mesh_mask(
                    verts=verts_pc.detach().cpu().numpy(),
                    faces=bind_faces_np,
                    K=K_np,
                    R=cam_r0,
                    t=cam_t0,
                    H=H,
                    W=W,
                    dilate_px=int(smpl_mask_dilate_px),
                    erode_px=int(smpl_mask_erode_px),
                )
                smpl_mask_cache[mesh_idx_pc] = (
                    torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                )

    for it in tqdm(range(iters), desc="Stage2 3DGS (learnable camera)"):
        opt.zero_grad(set_to_none=True)

        # Enable camera learning after warmup
        if learn_cam and it == cam_warmup_iters:
            _set_cam_trainable(True)

        # Select anchor frame (current frame to optimize).
        if ref_anchor == "last":
            fi = nF - 1
        elif frame_weights_t is None:
            fi = int(torch.randint(low=0, high=nF, size=(1,), generator=rng).item())
        else:
            fi = int(torch.multinomial(frame_weights_t, 1, replacement=True, generator=rng).item())

        # Build temporal reference set: current frame + previous frames with decay.
        if ref_window > 0:
            start = max(0, fi - ref_window)
            ctx_indices = list(range(start, fi + 1))
            raw_w = np.asarray([ref_decay ** (fi - fj) for fj in ctx_indices], dtype=np.float32)
            raw_w = raw_w / float(raw_w.sum() + 1e-8)
            ctx_weights = raw_w.tolist()
        else:
            ctx_indices = [fi]
            ctx_weights = [1.0]

        loss_value = 0.0
        for fj, wj in zip(ctx_indices, ctx_weights):
            gt = gts[fj]  # (3,H,W)

            # Camera pose for this reference frame
            R = axis_angle_to_rotmat(cam_w[fj])
            t = cam_t[fj]

            # Update bound xyz for this reference frame (dynamic binding)
            mesh_idx = None
            verts_t = None
            if binding_enabled:
                mesh_idx = int(bind_mesh_indices[fj])
                verts_t = _load_mesh_verts(mesh_idx)
                if bind_mask.any():
                    xyz_new = verts_t.index_select(0, bind_idx_t[bind_mask])
                    if bind_vertex_offset_t is not None:
                        xyz_new = xyz_new + bind_vertex_offset_t[bind_mask]
                    g.xyz[bind_mask] = xyz_new.detach()

            # Optional silhouette mask from bound SMPL mesh.
            smpl_mask = None
            if binding_enabled and bool(use_smpl_silhouette_mask) and bind_faces_np is not None:
                cache_key = int(mesh_idx) if (mesh_idx is not None and not learn_cam) else -1
                if cache_key >= 0 and cache_key in smpl_mask_cache:
                    smpl_mask = smpl_mask_cache[cache_key]
                else:
                    if verts_t is None:
                        raise RuntimeError("Silhouette mask requires bound verts, but verts_t is None.")
                    mask_np = _rasterize_mesh_mask(
                        verts=verts_t.detach().cpu().numpy(),
                        faces=bind_faces_np,
                        K=K_np,
                        R=R.detach().cpu().numpy().astype(np.float32),
                        t=t.detach().cpu().numpy().astype(np.float32),
                        H=H,
                        W=W,
                        dilate_px=int(smpl_mask_dilate_px),
                        erode_px=int(smpl_mask_erode_px),
                    )
                    smpl_mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                    if cache_key >= 0:
                        smpl_mask_cache[cache_key] = smpl_mask

            # Render
            need_alpha = bool(use_alpha_mask or float(smpl_alpha_leak_weight) > 0.0)
            if need_alpha:
                pred_rgb, pred_a = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=True)
                mask = (pred_a.detach() > alpha_threshold).float() if bool(use_alpha_mask) else None
            else:
                pred_rgb = render_gaussians_3d(g, K, R, t, H, W, bg=bg_rgb, return_alpha=False)
                pred_a = None
                mask = None

            # Loss: MSE + L1
            diff = pred_rgb - gt
            l2 = (diff * diff)
            l1 = diff.abs()

            if use_kpt_bbox_mask and kpt_boxes is not None and kpt_boxes[fj] is not None:
                x0, y0, x1, y1 = kpt_boxes[fj]
                m2d = torch.zeros((1, H, W), device=device)
                m2d[:, int(y0):int(y1) + 1, int(x0):int(x1) + 1] = 1.0
                mask = m2d if mask is None else (mask * m2d)
            if smpl_mask is not None:
                smpl_mask_for_color = smpl_mask
                if int(smpl_color_mask_erode_px) > 0:
                    smpl_mask_for_color = _erode_binary_mask_torch(
                        smpl_mask_for_color, int(smpl_color_mask_erode_px)
                    )
                if bool(smpl_mask_override_alpha):
                    mask = (
                        smpl_mask_for_color
                        if (mask is None or not use_kpt_bbox_mask)
                        else (mask * smpl_mask_for_color)
                    )
                else:
                    mask = smpl_mask_for_color if mask is None else (mask * smpl_mask_for_color)

            # Optional face emphasis (no hard mask, just higher weight)
            weight = None
            if face_boxes is not None and face_boxes[fj] is not None and float(face_weight) > 1.0:
                fx0, fy0, fx1, fy1 = face_boxes[fj]
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

            frame_loss = l2 + l1_weight * l1
            if pred_a is not None and smpl_mask is not None and float(smpl_alpha_leak_weight) > 0.0:
                outside = (1.0 - smpl_mask).clamp(0.0, 1.0)
                leak = (pred_a * outside).sum() / (outside.sum() + 1e-6)
                frame_loss = frame_loss + float(smpl_alpha_leak_weight) * leak
            if frame_loss_weights_t is not None:
                frame_loss = frame_loss * float(frame_loss_weights_t[fj].item())
            frame_loss = float(wj) * frame_loss
            loss_value += float(frame_loss.detach().cpu())
            frame_loss.backward()

        if learn_cam and cam_reg_weight > 0:
            # Keep camera near prior (prevents drift)
            reg = (cam_w ** 2).mean() + ((cam_t - cam_t_init) ** 2).mean()
            reg_loss = float(cam_reg_weight) * reg
            loss_value += float(reg_loss.detach().cpu())
            reg_loss.backward()
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
            "temporal_ref_window": temporal_ref_window,
            "temporal_ref_decay": temporal_ref_decay,
            "temporal_ref_anchor": temporal_ref_anchor,
            "bind_mesh_dir": str(bind_mesh_dir) if binding_enabled else None,
            "bind_mesh_indices": bind_mesh_indices if binding_enabled else None,
            "bind_vertices_npz": str(bind_vertices_npz) if (binding_enabled and bind_vertices_npz is not None) else None,
            "use_smpl_silhouette_mask": use_smpl_silhouette_mask,
            "smpl_mask_dilate_px": smpl_mask_dilate_px,
            "smpl_mask_erode_px": smpl_mask_erode_px,
            "smpl_color_mask_erode_px": smpl_color_mask_erode_px,
            "smpl_alpha_leak_weight": smpl_alpha_leak_weight,
            "smpl_mask_override_alpha": smpl_mask_override_alpha,
            "smpl_mask_precompute": smpl_mask_precompute,
            "smpl_mask_precompute_max_frames": smpl_mask_precompute_max_frames,
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
