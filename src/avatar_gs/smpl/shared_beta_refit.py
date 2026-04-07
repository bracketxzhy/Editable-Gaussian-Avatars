from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import smplx
except Exception:  # pragma: no cover
    smplx = None


@dataclass
class SharedBetaRefitConfig:
    enabled: bool = True
    min_align_iou: float = 0.99
    min_candidate_frames: int = 8
    use_exact_only: bool = True
    refit_iters: int = 90
    lr: float = 0.02
    vertex_sample_count: int = 4096
    joint_loss_weight: float = 1.0
    joint2d_loss_weight: float = 0.35
    bbox_loss_weight: float = 0.20
    vertex_loss_weight: float = 0.25
    pose_reg_weight: float = 0.005
    transl_reg_weight: float = 0.05
    device: str = "auto"


@dataclass
class SharedBetaRefitResult:
    records: list[dict[str, Any]]
    canonical_verts_cam: np.ndarray | None
    canonical_joints_cam: np.ndarray | None
    meta: dict[str, Any]


def _resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    raw = str(device).strip().lower()
    if raw in ("", "auto"):
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if raw == "cuda" and (not bool(torch.cuda.is_available())):
        return "cpu"
    return raw


def _reshape_like(flat: np.ndarray, ref: np.ndarray | None, fallback_dim: int) -> np.ndarray:
    x = np.asarray(flat, dtype=np.float32).reshape(-1)
    if ref is not None:
        ref_arr = np.asarray(ref)
        if ref_arr.size == x.size:
            return x.reshape(ref_arr.shape).astype(np.float32)
    return x.reshape(1, fallback_dim).astype(np.float32)


def _read_param(params: dict[str, Any], key: str, fallback_dim: int) -> np.ndarray:
    arr = np.asarray(params.get(key, np.zeros((1, fallback_dim), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if arr.size < fallback_dim:
        arr = np.pad(arr, (0, fallback_dim - arr.size))
    return arr[:fallback_dim].astype(np.float32)


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = np.clip(weights[order], 1e-8, None)
    cutoff = 0.5 * float(w.sum())
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    idx = int(np.clip(idx, 0, len(v) - 1))
    return float(v[idx])


def _estimate_shared_beta(
    records: list[dict[str, Any]],
    align_iou: np.ndarray | None,
    cfg: SharedBetaRefitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    betas: list[np.ndarray] = []
    exact_mask: list[bool] = []
    for rec in records:
        params = dict(rec.get("smplestx_params", {}))
        betas.append(np.asarray(params.get("betas", np.zeros((1, 10), dtype=np.float32)), dtype=np.float32).reshape(-1))
        exact_mask.append(str(rec.get("pred_source", "exact")).strip().lower() == "exact")
    beta_arr = np.asarray(betas, dtype=np.float32)
    n = int(beta_arr.shape[0])
    weights = np.ones((n,), dtype=np.float32)
    if align_iou is not None and align_iou.shape[0] == n:
        weights = np.clip(np.asarray(align_iou, dtype=np.float32).reshape(-1), 1e-3, None)
    mask = np.isfinite(beta_arr).all(axis=1)
    reason = "all_valid"
    if cfg.use_exact_only:
        exact_arr = np.asarray(exact_mask, dtype=bool)
        exact_valid = mask & exact_arr
        if int(exact_valid.sum()) >= max(1, int(cfg.min_candidate_frames)):
            mask = exact_valid
            reason = "exact_frames"
    if align_iou is not None and align_iou.shape[0] == n:
        high_q = mask & (np.asarray(align_iou, dtype=np.float32).reshape(-1) >= float(cfg.min_align_iou))
        if int(high_q.sum()) >= max(1, int(cfg.min_candidate_frames)):
            mask = high_q
            reason = f"align_iou>={float(cfg.min_align_iou):.3f}"
    if int(mask.sum()) <= 0:
        mask = np.isfinite(beta_arr).all(axis=1)
        reason = "fallback_all_valid"
    pick_idx = np.flatnonzero(mask).astype(np.int32)
    beta_pick = beta_arr[pick_idx]
    w_pick = weights[pick_idx]
    beta_shared = np.asarray(
        [_weighted_median_1d(beta_pick[:, d], w_pick) for d in range(int(beta_pick.shape[1]))],
        dtype=np.float32,
    )
    return beta_shared, pick_idx, w_pick.astype(np.float32), reason


def _create_model(assets_dir: Path, gender: str, device: str, model_type: str = "smplx"):
    if smplx is None or torch is None:
        raise RuntimeError("smplx and torch are required for shared-beta refit.")
    gender_use = str(gender).strip().lower()
    if gender_use not in {"neutral", "male", "female"}:
        gender_use = "neutral"
    return smplx.create(
        model_path=str(assets_dir),
        model_type=model_type,
        gender=gender_use,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to(device=device)


def _pick_vertex_indices(n_verts: int, max_count: int) -> np.ndarray:
    count = int(max(0, min(int(max_count), int(n_verts))))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count >= int(n_verts):
        return np.arange(int(n_verts), dtype=np.int64)
    return np.linspace(0, int(n_verts) - 1, num=count, dtype=np.int64)


def _project_points_torch(
    pts: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = pts[:, 2]
    valid = torch.isfinite(pts).all(dim=1) & (z.abs() > float(eps))
    z_safe = torch.where(valid, z, torch.ones_like(z))
    u = (pts[:, 0] / z_safe) * float(fx) + float(cx)
    v = (pts[:, 1] / z_safe) * float(fy) + float(cy)
    uv = torch.stack((u, v), dim=1)
    return uv, valid


def _sanitize_joints2d_np(value: object) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return np.asarray(arr[:, :3], dtype=np.float32)
    flat = arr.reshape(-1)
    usable = (int(flat.size) // 3) * 3
    if usable <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(flat[:usable], dtype=np.float32).reshape(-1, 3)


def _bbox_from_mask_np(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    mm = np.asarray(mask)
    if mm.ndim < 2:
        return None
    ys, xs = np.nonzero(mm > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    return np.asarray([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _extract_tpose_from_shared_beta(
    *,
    assets_dir: Path,
    beta_shared: np.ndarray,
    transl: np.ndarray,
    gender: str,
    device: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if smplx is None or torch is None:
        return None, None
    beta_shared = np.asarray(beta_shared, dtype=np.float32).reshape(-1)
    transl_np = np.asarray(transl, dtype=np.float32).reshape(3)
    for model_type in ("smplx", "smpl"):
        try:
            model = _create_model(assets_dir=assets_dir, gender=gender, device=device, model_type=model_type)
            num_betas = int(getattr(model, "num_betas", max(10, int(beta_shared.shape[0]))))
            betas = beta_shared
            if betas.shape[0] < num_betas:
                betas = np.pad(betas, (0, num_betas - betas.shape[0]))
            body_j = int(getattr(model, "NUM_BODY_JOINTS", 21 if model_type == "smplx" else 23))
            body_pose_dim = int(max(3, body_j * 3))
            betas_t = torch.from_numpy(betas[:num_betas].reshape(1, num_betas)).to(device=device, dtype=torch.float32)
            transl_t = torch.from_numpy(transl_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
            body_pose_base = torch.zeros((1, body_pose_dim), dtype=torch.float32, device=device)
            candidates = [body_pose_base.clone()]
            for ls, rs, lc, rc in ((15, 16, 13, 14), (14, 15, 12, 13), (16, 17, 14, 15), (13, 14, 11, 12)):
                for shoulder_deg, collar_deg in ((90.0, 20.0), (80.0, 15.0), (70.0, 10.0)):
                    bp = body_pose_base.clone()
                    for j_idx, deg in ((ls, +shoulder_deg), (rs, -shoulder_deg), (lc, +collar_deg), (rc, -collar_deg)):
                        k = int(j_idx) * 3 + 2
                        if 0 <= k < body_pose_dim:
                            bp[0, k] = float(np.deg2rad(deg))
                    candidates.append(bp)
            best_vertices = None
            best_joints = None
            best_span = -1.0
            for bp in candidates:
                kwargs = {
                    "betas": betas_t,
                    "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
                    "body_pose": bp,
                    "transl": transl_t,
                    "return_verts": True,
                }
                if model_type == "smplx":
                    num_expr = int(getattr(model, "num_expression_coeffs", 10))
                    kwargs.update(
                        {
                            "left_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
                            "right_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
                            "jaw_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
                            "leye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
                            "reye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
                            "expression": torch.zeros((1, num_expr), dtype=torch.float32, device=device),
                        }
                    )
                with torch.no_grad():
                    out = model(**kwargs)
                verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
                span_x = float(np.percentile(verts[:, 0], 99.0) - np.percentile(verts[:, 0], 1.0))
                if span_x > best_span:
                    best_span = span_x
                    best_vertices = verts
                    best_joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
            if best_vertices is not None and best_joints is not None:
                return best_vertices, best_joints
        except Exception:
            continue
    return None, None


def apply_shared_beta_refit(
    *,
    records: list[dict[str, Any]],
    assets_dir: Path,
    align_iou: np.ndarray | None = None,
    config: SharedBetaRefitConfig | None = None,
    person_masks_by_name: dict[str, np.ndarray] | None = None,
) -> SharedBetaRefitResult:
    cfg = SharedBetaRefitConfig() if config is None else config
    meta: dict[str, Any] = {"enabled": bool(cfg.enabled)}
    if (not bool(cfg.enabled)) or len(records) <= 0:
        meta["reason"] = "disabled_or_empty"
        return SharedBetaRefitResult(records=records, canonical_verts_cam=None, canonical_joints_cam=None, meta=meta)
    if torch is None or F is None or smplx is None:
        meta["reason"] = "missing_torch_or_smplx"
        return SharedBetaRefitResult(records=records, canonical_verts_cam=None, canonical_joints_cam=None, meta=meta)

    device = _resolve_device(cfg.device)
    beta_shared, candidate_idx, candidate_weights, candidate_reason = _estimate_shared_beta(records, align_iou, cfg)
    meta["device"] = device
    meta["shared_beta"] = beta_shared.astype(np.float32)
    meta["candidate_indices"] = candidate_idx.astype(np.int32)
    meta["candidate_weights"] = candidate_weights.astype(np.float32)
    meta["candidate_reason"] = str(candidate_reason)

    best_anchor_idx = int(candidate_idx[np.argmax(candidate_weights)]) if candidate_idx.size > 0 else int(len(records) - 1)
    meta["anchor_index"] = int(best_anchor_idx)

    model_cache: dict[tuple[str, str], Any] = {}
    vertex_losses = np.zeros((len(records),), dtype=np.float32)
    joint_losses = np.zeros((len(records),), dtype=np.float32)
    total_losses = np.zeros((len(records),), dtype=np.float32)

    for rec_idx, rec in enumerate(records):
        params = dict(rec.get("smplestx_params", {}))
        gender = str(rec.get("gender", "neutral")).strip().lower()
        model_key = ("smplx", gender)
        if model_key not in model_cache:
            model_cache[model_key] = _create_model(assets_dir=assets_dir, gender=gender, device=device, model_type="smplx")
        model = model_cache[model_key]

        num_betas = int(getattr(model, "num_betas", max(10, int(beta_shared.shape[0]))))
        num_expr = int(getattr(model, "num_expression_coeffs", 10))
        body_dim = int(getattr(model, "NUM_BODY_JOINTS", 21) * 3)
        betas = beta_shared
        if betas.shape[0] < num_betas:
            betas = np.pad(betas, (0, num_betas - betas.shape[0]))

        go0_np = _read_param(params, "global_orient", 3)
        bp0_np = _read_param(params, "body_pose", body_dim)
        tr0_np = np.asarray(rec.get("cam_t", params.get("transl", np.zeros((1, 3), dtype=np.float32))), dtype=np.float32).reshape(-1)
        if tr0_np.size < 3:
            tr0_np = np.pad(tr0_np, (0, 3 - tr0_np.size))
        tr0_np = tr0_np[:3].astype(np.float32)

        lh_np = _read_param(params, "left_hand_pose", 45)
        rh_np = _read_param(params, "right_hand_pose", 45)
        jaw_np = _read_param(params, "jaw_pose", 3)
        le_np = _read_param(params, "leye_pose", 3)
        re_np = _read_param(params, "reye_pose", 3)
        expr_np = _read_param(params, "expression", num_expr)

        go = torch.nn.Parameter(torch.from_numpy(go0_np.reshape(1, 3)).to(device=device, dtype=torch.float32))
        bp = torch.nn.Parameter(torch.from_numpy(bp0_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32))
        tr = torch.nn.Parameter(torch.from_numpy(tr0_np.reshape(1, 3)).to(device=device, dtype=torch.float32))

        go0_t = torch.from_numpy(go0_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
        bp0_t = torch.from_numpy(bp0_np.reshape(1, body_dim)).to(device=device, dtype=torch.float32)
        tr0_t = torch.from_numpy(tr0_np.reshape(1, 3)).to(device=device, dtype=torch.float32)
        betas_t = torch.from_numpy(betas[:num_betas].reshape(1, num_betas)).to(device=device, dtype=torch.float32)
        target_verts = torch.from_numpy(np.asarray(rec["verts_cam"], dtype=np.float32).reshape(-1, 3)).to(device=device, dtype=torch.float32)
        target_joints_np = np.asarray(rec.get("joints_cam", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        target_joints = torch.from_numpy(target_joints_np).to(device=device, dtype=torch.float32)
        target_joints2d_np = _sanitize_joints2d_np(rec.get("joints2d", None))
        target_joints2d = torch.from_numpy(target_joints2d_np[:, :2]).to(device=device, dtype=torch.float32) if target_joints2d_np.shape[0] > 0 else None
        target_joints2d_conf = (
            torch.from_numpy(target_joints2d_np[:, 2]).to(device=device, dtype=torch.float32)
            if target_joints2d_np.shape[0] > 0 and target_joints2d_np.shape[1] >= 3
            else None
        )
        v_idx_np = _pick_vertex_indices(int(target_verts.shape[0]), int(cfg.vertex_sample_count))
        v_idx = torch.from_numpy(v_idx_np).to(device=device, dtype=torch.long)
        focal_xy = np.asarray(rec.get("camera_focal", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        princpt_xy = np.asarray(rec.get("camera_princpt", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        have_intrinsics = focal_xy.size >= 2 and princpt_xy.size >= 2 and np.all(np.isfinite(focal_xy[:2])) and np.all(np.isfinite(princpt_xy[:2]))
        mask_bbox_np = None
        if isinstance(person_masks_by_name, dict):
            mask_bbox_np = _bbox_from_mask_np(person_masks_by_name.get(str(rec.get("frame_name", "")), None))
        target_bbox = (
            torch.from_numpy(mask_bbox_np).to(device=device, dtype=torch.float32)
            if mask_bbox_np is not None
            else None
        )

        fixed_kwargs = {
            "betas": betas_t,
            "left_hand_pose": torch.from_numpy(lh_np.reshape(1, 45)).to(device=device, dtype=torch.float32),
            "right_hand_pose": torch.from_numpy(rh_np.reshape(1, 45)).to(device=device, dtype=torch.float32),
            "jaw_pose": torch.from_numpy(jaw_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "leye_pose": torch.from_numpy(le_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "reye_pose": torch.from_numpy(re_np.reshape(1, 3)).to(device=device, dtype=torch.float32),
            "expression": torch.from_numpy(expr_np.reshape(1, num_expr)).to(device=device, dtype=torch.float32),
            "return_verts": True,
        }

        optimizer = torch.optim.Adam([go, bp, tr], lr=float(cfg.lr))
        last_loss_v = 0.0
        last_loss_j = 0.0
        last_loss = 0.0
        for it in range(max(1, int(cfg.refit_iters))):
            optimizer.zero_grad(set_to_none=True)
            out = model(
                global_orient=go,
                body_pose=bp,
                transl=tr,
                **fixed_kwargs,
            )
            verts = out.vertices[0]
            joints = out.joints[0]

            loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if int(v_idx.shape[0]) > 0:
                loss_v = F.smooth_l1_loss(verts.index_select(0, v_idx), target_verts.index_select(0, v_idx), reduction="mean")
                loss = loss + float(cfg.vertex_loss_weight) * loss_v
                last_loss_v = float(loss_v.detach().cpu().item())
            if int(target_joints.shape[0]) > 0:
                joint_n = min(int(joints.shape[0]), int(target_joints.shape[0]))
                if joint_n > 0:
                    loss_j = F.smooth_l1_loss(joints[:joint_n], target_joints[:joint_n], reduction="mean")
                    loss = loss + float(cfg.joint_loss_weight) * loss_j
                    last_loss_j = float(loss_j.detach().cpu().item())
            if have_intrinsics and target_joints2d is not None and int(target_joints2d.shape[0]) > 0 and float(cfg.joint2d_loss_weight) > 0.0:
                uv_j, valid_j = _project_points_torch(
                    joints,
                    fx=float(focal_xy[0]),
                    fy=float(focal_xy[1]),
                    cx=float(princpt_xy[0]),
                    cy=float(princpt_xy[1]),
                )
                joint_n2 = min(int(uv_j.shape[0]), int(target_joints2d.shape[0]))
                if joint_n2 > 0:
                    valid2 = valid_j[:joint_n2]
                    if target_joints2d_conf is not None and int(target_joints2d_conf.shape[0]) >= joint_n2:
                        valid2 = valid2 & (target_joints2d_conf[:joint_n2] > 0.05)
                    if bool(valid2.any().item()):
                        pred2 = uv_j[:joint_n2][valid2]
                        tgt2 = target_joints2d[:joint_n2][valid2]
                        if target_joints2d_conf is not None and int(target_joints2d_conf.shape[0]) >= joint_n2:
                            conf2 = target_joints2d_conf[:joint_n2][valid2].clamp(0.0, 1.0).unsqueeze(1)
                            loss_j2 = (F.smooth_l1_loss(pred2, tgt2, reduction="none") * conf2).mean()
                        else:
                            loss_j2 = F.smooth_l1_loss(pred2, tgt2, reduction="mean")
                        loss = loss + float(cfg.joint2d_loss_weight) * loss_j2
            if have_intrinsics and target_bbox is not None and float(cfg.bbox_loss_weight) > 0.0:
                uv_v, valid_v = _project_points_torch(
                    verts,
                    fx=float(focal_xy[0]),
                    fy=float(focal_xy[1]),
                    cx=float(princpt_xy[0]),
                    cy=float(princpt_xy[1]),
                )
                if bool(valid_v.any().item()):
                    uv_sel = uv_v[valid_v]
                    bbox_pred = torch.stack(
                        (
                            uv_sel[:, 0].amin(),
                            uv_sel[:, 1].amin(),
                            uv_sel[:, 0].amax(),
                            uv_sel[:, 1].amax(),
                        ),
                        dim=0,
                    )
                    loss_bbox = F.smooth_l1_loss(bbox_pred, target_bbox, reduction="mean")
                    loss = loss + float(cfg.bbox_loss_weight) * loss_bbox
            loss = loss + float(cfg.pose_reg_weight) * (
                F.mse_loss(go, go0_t, reduction="mean") + F.mse_loss(bp, bp0_t, reduction="mean")
            )
            loss = loss + float(cfg.transl_reg_weight) * F.mse_loss(tr, tr0_t, reduction="mean")
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if it == max(1, int(cfg.refit_iters)) // 2:
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * 0.35

        with torch.no_grad():
            out = model(
                global_orient=go,
                body_pose=bp,
                transl=tr,
                **fixed_kwargs,
            )
        verts_np = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        joints_np = out.joints[0].detach().cpu().numpy().astype(np.float32)
        transl_np = tr.detach().cpu().numpy().reshape(3).astype(np.float32)

        params["betas"] = _reshape_like(betas[:num_betas], params.get("betas", None), num_betas)
        params["global_orient"] = _reshape_like(go.detach().cpu().numpy(), params.get("global_orient", None), 3)
        params["body_pose"] = _reshape_like(bp.detach().cpu().numpy(), params.get("body_pose", None), body_dim)
        params["transl"] = _reshape_like(transl_np, params.get("transl", None), 3)
        rec["smplestx_params"] = params
        rec["cam_t"] = transl_np
        rec["verts_cam"] = verts_np
        rec["verts_local"] = np.asarray(verts_np - transl_np.reshape(1, 3), dtype=np.float32)
        rec["joints_cam"] = joints_np
        if have_intrinsics:
            uv_j, valid_j = _project_points_torch(
                out.joints[0],
                fx=float(focal_xy[0]),
                fy=float(focal_xy[1]),
                cx=float(princpt_xy[0]),
                cy=float(princpt_xy[1]),
            )
            rec["joints2d"] = np.concatenate(
                [
                    uv_j.detach().cpu().numpy().astype(np.float32),
                    valid_j.detach().cpu().numpy().astype(np.float32).reshape(-1, 1),
                ],
                axis=1,
            ).astype(np.float32)
        else:
            rec["joints2d"] = np.zeros((0, 3), dtype=np.float32)
        rec["shared_beta_refit_total_loss"] = float(last_loss)
        rec["shared_beta_refit_vertex_loss"] = float(last_loss_v)
        rec["shared_beta_refit_joint_loss"] = float(last_loss_j)
        vertex_losses[rec_idx] = float(last_loss_v)
        joint_losses[rec_idx] = float(last_loss_j)
        total_losses[rec_idx] = float(last_loss)

    anchor_rec = records[int(best_anchor_idx)]
    canonical_verts_cam, canonical_joints_cam = _extract_tpose_from_shared_beta(
        assets_dir=assets_dir,
        beta_shared=beta_shared,
        transl=np.asarray(anchor_rec.get("cam_t", np.zeros((3,), dtype=np.float32)), dtype=np.float32),
        gender=str(anchor_rec.get("gender", "neutral")),
        device=device,
    )

    meta["refit_vertex_loss"] = vertex_losses.astype(np.float32)
    meta["refit_joint_loss"] = joint_losses.astype(np.float32)
    meta["refit_total_loss"] = total_losses.astype(np.float32)
    meta["canonical_available"] = bool(canonical_verts_cam is not None and canonical_joints_cam is not None)

    return SharedBetaRefitResult(
        records=records,
        canonical_verts_cam=canonical_verts_cam,
        canonical_joints_cam=canonical_joints_cam,
        meta=meta,
    )

