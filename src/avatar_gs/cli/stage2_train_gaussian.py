from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything

from avatar_gs.gaussian.init_from_smpl import init_gaussians_from_mesh_3d
from avatar_gs.gaussian.train_loop import train_gaussians_3d_learnable_camera
from avatar_gs.data.h36m import resolve_h36m_image_paths


def main():
    parser = parse_args_base("Stage2: 3D Gaussian Splatting training with learnable camera (no COLMAP)")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    seed_everything(int(cfg.get("seed", 42)))

    project_root = Path(cfg.get("project_root", ".")).resolve()
    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("stage2_train_gaussian", str(log_dir / "stage2_train_gaussian.log"))

    stage1_dir = project_root / cfg["stage2"]["stage1_dir"]
    out_dir = ensure_dir(project_root / cfg["stage2"]["out_dir"])
    stage2_cfg = cfg["stage2"]

    if torch.cuda.is_available() and bool(stage2_cfg.get("enable_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(stage2_cfg.get("cudnn_benchmark", True))

    init_mesh_cfg = str(stage2_cfg.get("init_mesh_obj", "smpl_mesh_tpose.obj"))
    mesh_obj = Path(init_mesh_cfg)
    if not mesh_obj.is_absolute():
        mesh_obj = stage1_dir / mesh_obj
    if not mesh_obj.exists():
        fallback_mesh = None
        for cand in (stage1_dir / "smpl_mesh_tpose.obj", stage1_dir / "smpl_mesh.obj"):
            if cand.exists():
                fallback_mesh = cand
                break
        if fallback_mesh is None:
            raise FileNotFoundError(
                f"init mesh not found: {mesh_obj}. "
                f"Checked fallback: {stage1_dir / 'smpl_mesh_tpose.obj'} and {stage1_dir / 'smpl_mesh.obj'}."
            )
        mesh_obj = fallback_mesh

    cam = read_json(stage1_dir / "cameras.json")

    frames_dir = project_root / cfg["data"]["frames_dir"]
    n_frames = int(stage2_cfg.get("n_frames", 20))

    if "h36m" in cfg.get("data", {}):
        paths = resolve_h36m_image_paths(project_root=project_root, data_cfg=cfg["data"])
        # Use the actual image root to load files by name.
        frames_dir = paths[0].parent
        frames = [p.name for p in paths]
    else:
        frames = sorted([p.name for p in frames_dir.glob("*.jpg")])
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Frame selection strategy
    frame_select = str(stage2_cfg.get("frame_select", "first")).lower()
    if frame_select == "last":
        frames = [frames[-1]]
    elif frame_select == "last_k":
        k = max(1, int(n_frames))
        frames = frames[-k:]
    elif frame_select == "indices":
        indices = stage2_cfg.get("frame_indices", [])
        if not isinstance(indices, (list, tuple)) or len(indices) == 0:
            raise ValueError("frame_select=indices but frame_indices is empty.")
        picked = []
        for idx in indices:
            i = int(idx)
            if i < 0:
                i = len(frames) + i
            if i < 0 or i >= len(frames):
                raise IndexError(f"frame index out of range: {idx}")
            picked.append(frames[i])
        frames = picked
    else:
        # default: first N frames
        if n_frames > 0:
            frames = frames[:n_frames]

    logger.info(f"Mesh: {mesh_obj}")
    logger.info(f"Frames: {len(frames)} used for training")
    logger.info(f"Out dir: {out_dir}")

    camera_mode = str(cfg["stage2"].get("camera_mode", "learnable"))
    lr_cam = float(cfg["stage2"].get("lr_cam", 1e-3))
    cam_reg_weight = float(cfg["stage2"].get("cam_reg_weight", 0.0))
    if len(frames) == 1 and camera_mode.lower() != "fixed" and cam_reg_weight <= 0.0:
        logger.warning(
            "Single-frame training with learnable camera and cam_reg_weight<=0 "
            "is ill-posed; switching to fixed camera."
        )
        camera_mode = "fixed"
        lr_cam = 0.0

    # Frame sampling / loss weights (emphasize last frame)
    frame_weights = stage2_cfg.get("frame_weights", None)
    frame_loss_weights = stage2_cfg.get("frame_loss_weights", None)
    last_sample_w = stage2_cfg.get("last_frame_sample_weight", None)
    last_loss_w = stage2_cfg.get("last_frame_loss_weight", None)
    if frame_weights is None and last_sample_w is not None:
        frame_weights = [1.0] * len(frames)
        frame_weights[-1] = float(last_sample_w)
    if frame_loss_weights is None and last_loss_w is not None:
        frame_loss_weights = [1.0] * len(frames)
        frame_loss_weights[-1] = float(last_loss_w)

    g = init_gaussians_from_mesh_3d(
        mesh_obj=mesh_obj,
        n_keep=int(cfg["stage2"].get("n_gaussians", 3000)),
        seed=int(cfg.get("seed", 42)),
        n_bg=int(cfg["stage2"].get("n_gaussians_bg", 0)),
        bg_extent=tuple(cfg["stage2"].get("bg_extent", (3.0, 3.0, 0.6))),
        bg_center_z=float(cfg["stage2"].get("bg_center_z", 2.5)),
        bg_log_scale=float(cfg["stage2"].get("bg_log_scale", -3.5)),
        bg_opacity_logit=float(cfg["stage2"].get("bg_opacity_logit", -1.5)),
        bg_exclude_bbox_pad=float(cfg["stage2"].get("bg_exclude_bbox_pad", 0.0)),
        sample_on_surface=bool(cfg["stage2"].get("sample_on_surface", True)),
    )
    logger.info(f"Initialized gaussians: N={int(g.xyz.shape[0])}")

    # ---- Optional dynamic binding to SMPL mesh sequence ----
    bind_to_mesh = bool(cfg["stage2"].get("bind_to_mesh", False))
    bind_mesh_dir = None
    bind_mesh_indices = None
    bind_seq_npz_path = None
    bind_vertices_npz_path = None
    if bind_to_mesh:
        if g.bind_idx is None:
            raise RuntimeError("bind_to_mesh=true but gaussians have no bind_idx.")
        bind_mesh_dir_cfg = cfg["stage2"].get("bind_mesh_sequence_dir", str(stage1_dir / "smpl_mesh_sequence"))
        bind_mesh_dir = project_root / Path(bind_mesh_dir_cfg)
        if not bind_mesh_dir.exists():
            raise FileNotFoundError(
                f"bind_mesh_sequence_dir not found: {bind_mesh_dir}. "
                "Run stage1 with stage1.export_meshes=true to generate smpl_mesh_sequence."
            )

        seq_npz = stage1_dir / "smpl_params_sequence.npz"
        bind_seq_npz_path = seq_npz
        if not seq_npz.exists():
            raise FileNotFoundError(
                f"Missing {seq_npz}. Required to map frame names to mesh indices for binding."
            )
        verts_npz = stage1_dir / "smpl_vertices_sequence.npz"
        if verts_npz.exists():
            bind_vertices_npz_path = verts_npz
            logger.info(f"Using dynamic bind vertices from: {bind_vertices_npz_path}")
        else:
            logger.warning(
                f"Missing {verts_npz}; binding will fall back to OBJ sequence only."
            )
        seq = np.load(seq_npz, allow_pickle=True)
        seq_names = [str(x) for x in seq["frame_names"].tolist()]
        name_to_idx = {n: i for i, n in enumerate(seq_names)}

        missing = []
        bind_mesh_indices = []
        for fn in frames:
            if fn in name_to_idx:
                bind_mesh_indices.append(int(name_to_idx[fn]))
            else:
                missing.append(fn)

        if missing:
            if len(seq_names) == len(frames):
                logger.warning(
                    "Some frame names not found in smpl_params_sequence.npz; "
                    "falling back to sequential mesh indices."
                )
                bind_mesh_indices = list(range(len(frames)))
            else:
                raise RuntimeError(
                    f"Binding failed: {len(missing)} frames not found in smpl_params_sequence.npz. "
                    f"Example missing: {missing[0]}"
                )

        if bind_vertices_npz_path is not None:
            vseq = np.load(bind_vertices_npz_path, allow_pickle=True)
            if "verts_cam" in vseq.files:
                n_seq = int(np.asarray(vseq["verts_cam"]).shape[0])
                avail_set = set(range(n_seq))
                fallback_idx = max(n_seq - 1, 0)
                missing_seq = [mi for mi in bind_mesh_indices if mi not in avail_set]
                if len(missing_seq) > 0:
                    logger.warning(
                        "Some bind indices are out of range in smpl_vertices_sequence; "
                        f"falling back to frame {fallback_idx} for {len(missing_seq)} frames."
                    )
                    bind_mesh_indices = [mi if mi in avail_set else fallback_idx for mi in bind_mesh_indices]
            else:
                logger.warning(
                    f"{bind_vertices_npz_path} has no verts_cam; fallback to OBJ availability checks."
                )
                bind_vertices_npz_path = None
        if bind_vertices_npz_path is None:
            mesh_files = []
            for p in bind_mesh_dir.glob("*.obj"):
                stem = p.stem
                if stem.isdigit():
                    mesh_files.append(int(stem))
            if len(mesh_files) == 0:
                raise FileNotFoundError(f"No numeric OBJ meshes found in {bind_mesh_dir}")
            avail_set = set(mesh_files)
            fallback_idx = max(mesh_files)
            missing_mesh = [mi for mi in bind_mesh_indices if mi not in avail_set]
            if len(missing_mesh) > 0:
                logger.warning(
                    "Some bind mesh indices are missing from smpl_mesh_sequence; "
                    f"falling back to mesh {fallback_idx:06d}.obj for {len(missing_mesh)} frames."
                )
                bind_mesh_indices = [mi if mi in avail_set else fallback_idx for mi in bind_mesh_indices]

    cam_t_prior = tuple(cfg["stage2"].get("cam_t_prior", (0.0, 0.0, 0.0)))
    if (
        bind_to_mesh
        and bool(cfg["stage2"].get("bind_mesh_to_camera", False))
        and camera_mode.lower() == "fixed"
        and len(cam_t_prior) == 3
        and abs(float(cam_t_prior[0])) < 1e-8
        and abs(float(cam_t_prior[1])) < 1e-8
        and abs(float(cam_t_prior[2]) - 2.5) < 1e-8
    ):
        logger.warning(
            "cam_t_prior is [0,0,2.5] with bind_mesh_to_camera=true and fixed camera; "
            "this causes projection shrink/shift. Auto-correcting cam_t_prior to [0,0,0]."
        )
        cam_t_prior = (0.0, 0.0, 0.0)

    ckpt_path = out_dir / "gaussians3d.pt"
    train_gaussians_3d_learnable_camera(
        g=g,
        frames_dir=frames_dir,
        frames_list=frames,
        K_list=cam["K"],
        out_ckpt=ckpt_path,
        iters=int(cfg["stage2"].get("iters", 2000)),
        lr_gauss=float(cfg["stage2"].get("lr_gauss", 1e-2)),
        lr_cam=lr_cam,
        bg_rgb=tuple(cfg["stage2"].get("bg_rgb", (1.0, 1.0, 1.0))),
        cam_warmup_iters=int(cfg["stage2"].get("cam_warmup_iters", 5000)),
        l1_weight=float(cfg["stage2"].get("l1_weight", 0.25)),
        use_alpha_mask=bool(cfg["stage2"].get("use_alpha_mask", True)),
        alpha_threshold=float(cfg["stage2"].get("alpha_threshold", 0.15)),
        camera_mode=camera_mode,
        cam_t_prior=cam_t_prior,
        cam_reg_weight=cam_reg_weight,
        keypoints_json=cfg["stage2"].get("keypoints_json", str(stage1_dir / "keypoints.json")),
        kpt_conf_min=float(cfg["stage2"].get("kpt_conf_min", 0.2)),
        kpt_box_pad=float(cfg["stage2"].get("kpt_box_pad", 0.15)),
        use_kpt_bbox_mask=bool(cfg["stage2"].get("use_kpt_bbox_mask", True)),
        face_weight=float(cfg["stage2"].get("face_weight", 1.0)),
        face_conf_min=float(cfg["stage2"].get("face_conf_min", 0.3)),
        face_box_pad=float(cfg["stage2"].get("face_box_pad", 0.2)),
        frame_weights=frame_weights,
        frame_loss_weights=frame_loss_weights,
        temporal_ref_window=int(cfg["stage2"].get("temporal_ref_window", 0)),
        temporal_ref_decay=float(cfg["stage2"].get("temporal_ref_decay", 0.85)),
        temporal_ref_anchor=str(cfg["stage2"].get("temporal_ref_anchor", "sample")),
        bind_mesh_dir=bind_mesh_dir,
        bind_vertex_indices=g.bind_idx,
        bind_mesh_indices=bind_mesh_indices,
        bind_cache_size=int(cfg["stage2"].get("bind_cache_size", 8)),
        bind_seq_npz=str(bind_seq_npz_path) if bind_seq_npz_path is not None else None,
        bind_vertices_npz=str(bind_vertices_npz_path) if bind_vertices_npz_path is not None else None,
        bind_mesh_to_camera=bool(cfg["stage2"].get("bind_mesh_to_camera", False)),
        bind_inverse_export_upright=bool(cfg["stage2"].get("bind_inverse_export_upright", True)),
        bind_mesh_offset=tuple(cfg["stage2"].get("bind_mesh_offset", (0.0, 0.0, 0.0))),
        use_smpl_silhouette_mask=bool(cfg["stage2"].get("use_smpl_silhouette_mask", False)),
        smpl_mask_dilate_px=int(cfg["stage2"].get("smpl_mask_dilate_px", 0)),
        smpl_mask_erode_px=int(cfg["stage2"].get("smpl_mask_erode_px", 0)),
        smpl_color_mask_erode_px=int(cfg["stage2"].get("smpl_color_mask_erode_px", 0)),
        smpl_alpha_leak_weight=float(cfg["stage2"].get("smpl_alpha_leak_weight", 0.0)),
        smpl_mask_override_alpha=bool(cfg["stage2"].get("smpl_mask_override_alpha", True)),
        smpl_mask_precompute=bool(cfg["stage2"].get("smpl_mask_precompute", True)),
        smpl_mask_precompute_max_frames=int(cfg["stage2"].get("smpl_mask_precompute_max_frames", 64)),
        bg_grad_scale=float(cfg["stage2"].get("bg_grad_scale", 1.0)),
        bg_opacity_logit_max=cfg["stage2"].get("bg_opacity_logit_max", None),
        bg_log_scale_min=cfg["stage2"].get("bg_log_scale_min", None),
        bg_log_scale_max=cfg["stage2"].get("bg_log_scale_max", None),
        fg_opacity_logit_max=cfg["stage2"].get("fg_opacity_logit_max", None),
        fg_log_scale_min=cfg["stage2"].get("fg_log_scale_min", None),
        fg_log_scale_max=cfg["stage2"].get("fg_log_scale_max", None),
    )

    logger.info(f"Saved checkpoint: {ckpt_path}")
    save_yaml(cfg, log_dir / "stage2_train_gaussian_resolved.yaml")


if __name__ == "__main__":
    main()
