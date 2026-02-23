from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

try:
    import smplx
except Exception:
    smplx = None


def _parse_frame_idx_from_name(frame_name: str) -> int | None:
    stem = Path(str(frame_name)).stem.strip().lower()
    if stem.isdigit():
        return int(stem)
    if stem.startswith("frame_") and stem[6:].isdigit():
        return int(stem[6:])
    return None


def _pick_param_key(npz_obj, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in npz_obj.files:
            return k
    return None


def _resolve_npz_row_index(npz_obj, frame_index: int = -1, frame_name: str = "") -> int:
    n = 0
    if "frame_names" in npz_obj.files:
        try:
            n = int(np.asarray(npz_obj["frame_names"]).reshape(-1).shape[0])
        except Exception:
            n = 0
    if n <= 0:
        for k in ("global_orient", "body_pose", "pose_body", "verts_cam"):
            if k in npz_obj.files:
                arr = np.asarray(npz_obj[k])
                if arr.ndim >= 2:
                    n = int(arr.shape[0])
                    break
    if n <= 0:
        return 0

    # Explicit frame_index has higher priority than frame_name.
    # This avoids unintentionally overriding user-requested deform frame with render-selected frame.
    if int(frame_index) >= 0:
        fi = int(frame_index)
        if fi < n:
            return fi
        if "frame_names" in npz_obj.files:
            names = [str(x) for x in np.asarray(npz_obj["frame_names"]).reshape(-1).tolist()]
            for i, nm in enumerate(names):
                p = _parse_frame_idx_from_name(nm)
                if p is not None and p == fi:
                    return int(i)

    if frame_name:
        req = str(frame_name).strip()
        if "frame_names" in npz_obj.files:
            names = [str(x) for x in np.asarray(npz_obj["frame_names"]).reshape(-1).tolist()]
            if req in names:
                return int(names.index(req))
            req_stem = Path(req).stem
            for i, nm in enumerate(names):
                if Path(str(nm)).stem == req_stem:
                    return int(i)

    return max(0, n - 1)


def _read_param_row(npz_obj, key: str, row_idx: int) -> np.ndarray | None:
    if key not in npz_obj.files:
        return None
    arr = np.asarray(npz_obj[key], dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    ridx = int(np.clip(int(row_idx), 0, max(int(arr.shape[0]) - 1, 0)))
    return np.asarray(arr[ridx], dtype=np.float32)


def _fit_param_dim(vec: np.ndarray | None, dim: int, default: float = 0.0) -> np.ndarray:
    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if vec is None:
        return np.full((dim,), float(default), dtype=np.float32)
    x = np.asarray(vec, dtype=np.float32).reshape(-1)
    if x.shape[0] < dim:
        x = np.pad(x, (0, dim - x.shape[0]), constant_values=float(default))
    elif x.shape[0] > dim:
        x = x[:dim]
    return x.astype(np.float32)


def _match_direct_space_to_reference(
    verts: np.ndarray,
    joints: np.ndarray,
    ref_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match deform vertices/joints to a reference camera-space mesh using
    orientation candidates that preserve handedness.
    """
    v = np.asarray(verts, dtype=np.float32).copy()
    j = np.asarray(joints, dtype=np.float32).copy() if joints is not None and joints.size > 0 else np.zeros((0, 3), dtype=np.float32)
    ref = np.asarray(ref_verts, dtype=np.float32)

    if v.ndim != 2 or v.shape[1] != 3 or ref.ndim != 2 or ref.shape[1] != 3:
        return v.astype(np.float32), j.astype(np.float32), np.ones((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    cand_signs = [
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 1.0, -1.0], dtype=np.float32),
        np.asarray([1.0, -1.0, -1.0], dtype=np.float32),
        np.asarray([-1.0, -1.0, 1.0], dtype=np.float32),
    ]

    if v.shape[0] == ref.shape[0]:
        v_idx = None
        r_idx = None
    else:
        n = int(min(v.shape[0], ref.shape[0]))
        if n <= 0:
            return v.astype(np.float32), j.astype(np.float32), np.ones((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
        v_idx = np.linspace(0, v.shape[0] - 1, n).round().astype(np.int64)
        r_idx = np.linspace(0, ref.shape[0] - 1, n).round().astype(np.int64)

    best_sign = cand_signs[0]
    best_delta = np.zeros((3,), dtype=np.float32)
    best_err = float("inf")

    for sign in cand_signs:
        vv = v * sign.reshape(1, 3)
        if v_idx is None:
            vv_cmp = vv
            rr_cmp = ref
        else:
            vv_cmp = vv[v_idx]
            rr_cmp = ref[r_idx]
        delta = (rr_cmp.mean(axis=0) - vv_cmp.mean(axis=0)).astype(np.float32)
        err = float(np.mean(np.linalg.norm((vv_cmp + delta.reshape(1, 3)) - rr_cmp, axis=1)))
        if err < best_err:
            best_err = err
            best_sign = sign.copy()
            best_delta = delta.copy()

    v = v * best_sign.reshape(1, 3) + best_delta.reshape(1, 3)
    if j.size > 0:
        j = j * best_sign.reshape(1, 3) + best_delta.reshape(1, 3)

    return v.astype(np.float32), j.astype(np.float32), best_sign.astype(np.float32), best_delta.astype(np.float32)


def load_deform_smplx_vertices_from_npz(
    *,
    params_npz_path: Path,
    frame_index: int,
    frame_name: str,
    model_type_default: str,
    gender_default: str,
    model_root_cfg: str,
    project_root: Path,
    assets_dir: Path,
    ref_vertices_npz_path: Path | None = None,
    use_standard_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not params_npz_path.exists():
        raise FileNotFoundError(f"deform_smplx_params_npz not found: {params_npz_path}")

    z = np.load(params_npz_path, allow_pickle=True)
    row_idx = _resolve_npz_row_index(z, frame_index=frame_index, frame_name=frame_name)

    def _load_ref_verts_row_external() -> np.ndarray | None:
        p_ref = ref_vertices_npz_path
        if p_ref is None:
            return None
        p_ref = Path(p_ref)
        if not p_ref.exists():
            return None
        try:
            rz = np.load(p_ref, allow_pickle=True)
        except Exception:
            return None
        if "verts_cam" not in rz.files:
            return None
        vc = np.asarray(rz["verts_cam"], dtype=np.float32)
        if vc.ndim != 3 or vc.shape[2] != 3 or vc.shape[0] <= 0:
            return None

        ridx = None
        req_name = str(frame_name).strip()
        if req_name and ("frame_names" in rz.files):
            try:
                names = [str(x) for x in np.asarray(rz["frame_names"]).reshape(-1).tolist()]
                if req_name in names:
                    ridx = int(names.index(req_name))
                else:
                    req_stem = Path(req_name).stem
                    for ii, nm in enumerate(names):
                        if Path(str(nm)).stem == req_stem:
                            ridx = int(ii)
                            break
            except Exception:
                ridx = None

        if ridx is None:
            idx_try = int(frame_index)
            if idx_try < 0:
                idx_try = int(row_idx)
            ridx = int(np.clip(idx_try, 0, int(vc.shape[0]) - 1))

        return np.asarray(vc[ridx], dtype=np.float32)

    def _direct_joints_canonical() -> np.ndarray:
        key = "joints_canonical_cam" if "joints_canonical_cam" in z.files else ("joints_tpose_cam" if "joints_tpose_cam" in z.files else "")
        if not key:
            return np.zeros((0, 3), dtype=np.float32)
        jt = np.asarray(z[key], dtype=np.float32)
        if jt.ndim == 3 and jt.shape[0] == 1:
            jt = jt[0]
        if jt.ndim == 2 and jt.shape[1] == 3:
            return jt.astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def _direct_joints_row() -> np.ndarray:
        if "joints_cam" not in z.files:
            return np.zeros((0, 3), dtype=np.float32)
        jc = np.asarray(z["joints_cam"], dtype=np.float32)
        if jc.ndim == 3 and jc.shape[2] == 3 and jc.shape[0] > 0:
            ridx = int(np.clip(int(row_idx), 0, int(jc.shape[0]) - 1))
            return np.asarray(jc[ridx], dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def _load_direct_vertices() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
        vc_all = None
        vc_row = None
        if "verts_cam" in z.files:
            vc_all = np.asarray(z["verts_cam"], dtype=np.float32)
            if vc_all.ndim == 3 and vc_all.shape[2] == 3 and vc_all.shape[0] > 0:
                ridx = int(np.clip(int(row_idx), 0, int(vc_all.shape[0]) - 1))
                vc_row = np.asarray(vc_all[ridx], dtype=np.float32)

        if vc_row is None:
            vc_row = _load_ref_verts_row_external()

        canonical_key = "verts_canonical_cam" if "verts_canonical_cam" in z.files else ("verts_tpose_cam" if "verts_tpose_cam" in z.files else "")
        if bool(use_standard_pose) and canonical_key:
            vv = np.asarray(z[canonical_key], dtype=np.float32)
            if vv.ndim == 2 and vv.shape[1] == 3:
                jj = _direct_joints_canonical()
                meta = {
                    "row_idx": int(row_idx),
                    "model_type": "direct",
                    "gender": "direct",
                    "model_root": "direct_npz",
                    "params_npz": str(params_npz_path),
                    "source": canonical_key,
                }
                if vc_row is not None:
                    vv_m, jj_m, sign, delta = _match_direct_space_to_reference(vv, jj, vc_row)
                    meta["space_matched_to"] = "verts_cam"
                    meta["space_match_sign"] = sign.tolist()
                    meta["space_match_delta"] = delta.tolist()
                    return vv_m, jj_m, meta
                return vv, jj, meta

        if vc_row is not None:
            return vc_row, _direct_joints_row(), {
                "row_idx": int(np.clip(int(row_idx), 0, int(vc_row.shape[0]) - 1)) if vc_all is not None else int(row_idx),
                "model_type": "direct",
                "gender": "direct",
                "model_root": "direct_npz",
                "params_npz": str(params_npz_path),
                "source": "verts_cam",
            }

        return None

    param_keys = [
        "global_orient", "root_orient", "body_pose", "pose_body",
        "betas", "shape", "transl", "translation", "cam_t",
    ]
    has_model_params = any(k in z.files for k in param_keys)

    if not has_model_params:
        direct = _load_direct_vertices()
        if direct is not None:
            return direct
        raise RuntimeError(
            "deform_smplx_params_npz has no SMPL-X parameter arrays and no usable direct verts "
            "(expected one of: verts_canonical_cam / verts_tpose_cam / verts_cam)."
        )

    if smplx is None:
        raise RuntimeError("smplx is not installed, cannot use deform_source=smplx_params")

    model_type = str(model_type_default).strip().lower() or "smplx"
    gender = str(gender_default).strip().lower() or "neutral"
    if "model_type" in z.files:
        mt_arr = np.asarray(z["model_type"]).reshape(-1)
        if mt_arr.size > 0:
            model_type = str(mt_arr[min(row_idx, mt_arr.size - 1)]).strip().lower() or model_type
    if "gender" in z.files:
        gd_arr = np.asarray(z["gender"]).reshape(-1)
        if gd_arr.size > 0:
            gender = str(gd_arr[min(row_idx, gd_arr.size - 1)]).strip().lower() or gender
    if model_type not in ("smpl", "smplx"):
        model_type = "smplx"
    if gender not in ("neutral", "male", "female"):
        gender = "neutral"

    if str(model_root_cfg).strip():
        cfg_path = Path(str(model_root_cfg))
        if cfg_path.is_absolute():
            model_root = cfg_path
        else:
            proj_candidate = project_root / cfg_path
            if proj_candidate.exists():
                model_root = proj_candidate
            else:
                model_root = assets_dir / cfg_path
    else:
        model_root = assets_dir
    if not model_root.exists():
        raise FileNotFoundError(f"SMPL model root not found for deform_source=smplx_params: {model_root}")

    if model_type == "smplx":
        direct_file = model_root / "SMPLX_NEUTRAL.npz"
    else:
        direct_file = model_root / "SMPL_NEUTRAL.npz"
    model_root_for_create = model_root.parent if direct_file.exists() else model_root

    model = smplx.create(
        model_path=str(model_root_for_create),
        model_type=model_type,
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device="cpu")

    body_j = int(getattr(model, "NUM_BODY_JOINTS", 21 if model_type == "smplx" else 23))
    body_pose_dim = int(max(3, body_j * 3))
    num_betas = int(getattr(model, "num_betas", 10))

    go = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["global_orient", "root_orient"]) or "", row_idx), 3)
    bp = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["body_pose", "pose_body"]) or "", row_idx), body_pose_dim)
    bt = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["betas", "shape"]) or "", row_idx), num_betas)
    tr = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["transl", "translation", "cam_t"]) or "", row_idx), 3)

    kwargs = {
        "betas": torch.from_numpy(bt.reshape(1, num_betas)).to(dtype=torch.float32),
        "global_orient": torch.from_numpy(go.reshape(1, 3)).to(dtype=torch.float32),
        "body_pose": torch.from_numpy(bp.reshape(1, body_pose_dim)).to(dtype=torch.float32),
        "transl": torch.from_numpy(tr.reshape(1, 3)).to(dtype=torch.float32),
        "return_verts": True,
    }

    if model_type == "smplx":
        lh = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["left_hand_pose"]) or "", row_idx), 45)
        rh = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["right_hand_pose"]) or "", row_idx), 45)
        jaw = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["jaw_pose"]) or "", row_idx), 3)
        le = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["leye_pose"]) or "", row_idx), 3)
        re = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["reye_pose"]) or "", row_idx), 3)
        num_expr = int(getattr(model, "num_expression_coeffs", 10))
        expr = _fit_param_dim(_read_param_row(z, _pick_param_key(z, ["expression", "expr"]) or "", row_idx), num_expr)
        kwargs.update(
            {
                "left_hand_pose": torch.from_numpy(lh.reshape(1, 45)).to(dtype=torch.float32),
                "right_hand_pose": torch.from_numpy(rh.reshape(1, 45)).to(dtype=torch.float32),
                "jaw_pose": torch.from_numpy(jaw.reshape(1, 3)).to(dtype=torch.float32),
                "leye_pose": torch.from_numpy(le.reshape(1, 3)).to(dtype=torch.float32),
                "reye_pose": torch.from_numpy(re.reshape(1, 3)).to(dtype=torch.float32),
                "expression": torch.from_numpy(expr.reshape(1, num_expr)).to(dtype=torch.float32),
            }
        )

    with torch.no_grad():
        out = model(**kwargs)
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)

    meta = {
        "row_idx": int(row_idx),
        "model_type": model_type,
        "gender": gender,
        "model_root": str(model_root),
        "params_npz": str(params_npz_path),
        "source": "smplx_rebuild",
    }

    ref_row = _load_ref_verts_row_external()
    if ref_row is not None:
        verts_m, joints_m, sign, delta = _match_direct_space_to_reference(verts, joints, ref_row)
        meta["space_matched_to"] = "verts_cam"
        meta["space_match_sign"] = sign.tolist()
        meta["space_match_delta"] = delta.tolist()
        return verts_m, joints_m, meta

    return verts, joints, meta
