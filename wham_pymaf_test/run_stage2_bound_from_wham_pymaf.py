from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _resolve_path(project_root: Path, path_like: str | Path) -> Path:
    p = Path(str(path_like))
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def _load_obj_vertices(obj_path: Path) -> np.ndarray:
    verts = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            toks = line.strip().split()
            if len(toks) < 4:
                continue
            verts.append([float(toks[1]), float(toks[2]), float(toks[3])])
    if len(verts) <= 0:
        raise RuntimeError(f"No vertices found in OBJ: {obj_path}")
    return np.asarray(verts, dtype=np.float32)


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_meta_path(final_dir: Path, path_like: str | Path) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p.resolve()
    cand = (final_dir / p).resolve()
    if cand.exists():
        return cand
    return p.resolve()


def _discover_frame_name(final_dir: Path, fallback_frames_dir: Path) -> str:
    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    if coord_meta is not None and str(coord_meta.get("frame_name", "")).strip():
        return str(coord_meta["frame_name"]).strip()

    summary = _load_json_if_exists(final_dir / "run_summary.json")
    if summary is not None and str(summary.get("last_frame_name", "")).strip():
        return str(summary["last_frame_name"]).strip()

    gt_cands = sorted(final_dir.glob("last_frame_gt_*.jpg"))
    if len(gt_cands) > 0:
        name = gt_cands[-1].name
        prefix = "last_frame_gt_"
        if name.startswith(prefix):
            return name[len(prefix):]

    frames = sorted(fallback_frames_dir.glob("*.jpg"))
    if len(frames) <= 0:
        raise RuntimeError(f"Cannot discover frame_name: no jpg found in {fallback_frames_dir}")
    return frames[-1].name


def _resolve_cam_obj(final_dir: Path) -> Path:
    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    if coord_meta is not None:
        p = coord_meta.get("camera_world_obj", None)
        if p is not None:
            cand = Path(str(p))
            if cand.exists():
                return cand.resolve()

    cands = sorted(final_dir.glob("last_frame_*_cam.obj"))
    if len(cands) > 0:
        return cands[0].resolve()

    raise FileNotFoundError(
        "Camera/world OBJ not found. Expected `last_frame_*_cam.obj` under "
        f"{final_dir}. Re-run `wham_pymaf_test/run_wham_pymaf_test.py` with updated exporter."
    )


def _resolve_legacy_local_obj(final_dir: Path) -> Path:
    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    if coord_meta is not None:
        p = coord_meta.get("pose_local_obj", None)
        if p is not None:
            cand = _resolve_meta_path(final_dir, p)
            if cand.exists():
                return cand
    summary = _load_json_if_exists(final_dir / "run_summary.json")
    if summary is not None:
        p = summary.get("last_frame_pose_local_obj", None)
        if p is not None:
            cand = _resolve_meta_path(final_dir, p)
            if cand.exists():
                return cand
        p = summary.get("last_frame_obj", None)
        if p is not None:
            cand = _resolve_meta_path(final_dir, p)
            if cand.exists():
                return cand
    cands = []
    for p in final_dir.glob("last_frame*.obj"):
        n = p.name.lower()
        if "_cam.obj" in n:
            continue
        cands.append(p)
    cands = sorted(cands)
    if len(cands) > 0:
        return cands[0].resolve()
    raise FileNotFoundError(
        f"Cannot find legacy/local OBJ in {final_dir}. "
        "Expected last_frame_pose_local.obj or last_frame_*.obj"
    )


def _load_pymaf_output_cpu_safe(pred_pkl: Path) -> dict:
    try:
        import joblib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Need joblib to load PyMAF output: {pred_pkl}. "
            f"Import error: {e}"
        )
    try:
        return joblib.load(pred_pkl)
    except Exception as e:
        msg = str(e)
        if ("CUDA" not in msg) and ("cuda" not in msg):
            raise
        try:
            import io
            import torch.storage  # type: ignore
            import torch  # type: ignore
        except Exception:
            raise
        _old = torch.storage._load_from_bytes
        try:
            torch.storage._load_from_bytes = lambda b: torch.load(  # type: ignore
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
            return joblib.load(pred_pkl)
        finally:
            torch.storage._load_from_bytes = _old


def _resolve_cam_t_from_outputs(final_dir: Path, frame_name: str) -> tuple[np.ndarray, str]:
    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    if coord_meta is not None and coord_meta.get("cam_t", None) is not None:
        cam_t = np.asarray(coord_meta["cam_t"], dtype=np.float32).reshape(3,)
        return cam_t, "coord_meta"

    pymaf_dir = final_dir.parent / "pymaf" / "selected_frames"
    pred_pkl = pymaf_dir / "output.pkl"
    meta_json = pymaf_dir / "selected_frames_meta.json"
    if (not pred_pkl.exists()) or (not meta_json.exists()):
        raise FileNotFoundError(
            "Cannot recover cam_t automatically: missing "
            f"{pred_pkl} or {meta_json}."
        )

    pred_results = _load_pymaf_output_cpu_safe(pred_pkl)
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)
    sel_files = [str(x) for x in selected_meta.get("selected_files", [])]
    if len(sel_files) <= 0:
        raise RuntimeError(f"selected_frames_meta has empty selected_files: {meta_json}")
    if frame_name in sel_files:
        local_idx = int(sel_files.index(frame_name))
    else:
        local_idx = int(len(sel_files) - 1)
    frame_ids = np.asarray(pred_results["frame_ids"], dtype=np.int64).reshape(-1)
    cand = np.where(frame_ids == local_idx)[0]
    if cand.size <= 0:
        idx = int(np.argmax(frame_ids))
    else:
        idx = int(cand[0])
    cam_t = np.asarray(pred_results["orig_cam_t"][idx], dtype=np.float32).reshape(3,)
    return cam_t, "pymaf_output"


def _load_pymaf_frame_record(final_dir: Path, frame_name: str) -> tuple[dict, int]:
    pymaf_dir = final_dir.parent / "pymaf" / "selected_frames"
    pred_pkl = pymaf_dir / "output.pkl"
    meta_json = pymaf_dir / "selected_frames_meta.json"
    if (not pred_pkl.exists()) or (not meta_json.exists()):
        raise FileNotFoundError(
            "Missing PyMAF selected-frame outputs for bind asset generation: "
            f"{pred_pkl} or {meta_json}"
        )
    pred_results = _load_pymaf_output_cpu_safe(pred_pkl)
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    sel_files = [str(x) for x in selected_meta.get("selected_files", [])]
    if len(sel_files) <= 0:
        raise RuntimeError(f"selected_frames_meta has empty selected_files: {meta_json}")
    if frame_name in sel_files:
        local_idx = int(sel_files.index(frame_name))
    else:
        local_idx = int(len(sel_files) - 1)

    frame_ids = np.asarray(pred_results["frame_ids"], dtype=np.int64).reshape(-1)
    cand = np.where(frame_ids == local_idx)[0]
    if cand.size <= 0:
        idx = int(np.argmax(frame_ids))
    else:
        idx = int(cand[0])
    return pred_results, idx


def _write_obj_replace_vertices(template_obj: Path, out_obj: Path, verts: np.ndarray) -> None:
    lines = template_obj.read_text(encoding="utf-8").splitlines()
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    vi = 0
    out_lines: list[str] = []
    n_verts = int(np.asarray(verts).shape[0])
    for ln in lines:
        if ln.startswith("v "):
            if vi >= n_verts:
                raise RuntimeError(
                    f"Vertex count mismatch while writing {out_obj}: template has more v-lines than verts ({n_verts})."
                )
            v = verts[vi]
            out_lines.append(f"v {float(v[0]):.7f} {float(v[1]):.7f} {float(v[2]):.7f}")
            vi += 1
        else:
            out_lines.append(ln)
    if vi != n_verts:
        raise RuntimeError(
            f"Vertex count mismatch while writing {out_obj}: template has {vi} v-lines, new verts has {n_verts}."
        )
    out_obj.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _resolve_or_make_cam_obj(final_dir: Path, frame_name: str) -> Path:
    try:
        return _resolve_cam_obj(final_dir)
    except FileNotFoundError:
        pass

    local_obj = _resolve_legacy_local_obj(final_dir)
    verts_local = _load_obj_vertices(local_obj)
    cam_t, cam_t_source = _resolve_cam_t_from_outputs(final_dir=final_dir, frame_name=frame_name)
    verts_cam = verts_local + cam_t.reshape(1, 3)

    summary = _load_json_if_exists(final_dir / "run_summary.json") or {}
    model_type = str(summary.get("model_type", "")).strip().lower()
    if model_type not in ("smpl", "smplx"):
        model_type = "smplx" if "smplx" in local_obj.name.lower() else "smpl"

    pose_local_obj = final_dir / "last_frame_pose_local.obj"
    if not pose_local_obj.exists():
        shutil.copy2(local_obj, pose_local_obj)
    cam_obj = final_dir / f"last_frame_{model_type}_cam.obj"
    _write_obj_replace_vertices(pose_local_obj, cam_obj, verts_cam)

    coord_meta_path = final_dir / "last_frame_coord_meta.json"
    coord_meta = _load_json_if_exists(coord_meta_path) or {}
    coord_meta.update(
        {
            "frame_name": frame_name,
            "model_type": model_type,
            "cam_t": cam_t.astype(np.float32).reshape(3).tolist(),
            "pose_local_obj": str(pose_local_obj),
            "camera_world_obj": str(cam_obj),
            "legacy_obj": str(local_obj),
            "cam_t_source": cam_t_source,
            "coord_system": {
                "pose_local_obj": "local pose mesh (no camera translation)",
                "camera_world_obj": "camera/world metric mesh (verts + cam_t)",
            },
        }
    )
    with coord_meta_path.open("w", encoding="utf-8") as f:
        json.dump(coord_meta, f, indent=2)
    print(f"[Compat] Generated camera/world OBJ from legacy outputs: {cam_obj}")
    print(f"[Compat] Wrote coordinate metadata: {coord_meta_path}")
    return cam_obj


def _validate_coordinate_contract(
    *,
    final_dir: Path,
    cam_obj: Path,
    z_median_min: float,
    coord_tol: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    verts_cam = _load_obj_vertices(cam_obj)
    z_median = float(np.median(verts_cam[:, 2]))
    if z_median <= float(z_median_min):
        raise RuntimeError(
            "Camera/world OBJ sanity check failed: median z="
            f"{z_median:.4f} <= {float(z_median_min):.4f}. "
            "This usually means a local-space mesh was used by mistake."
        )

    diag = {
        "camera_world_obj": str(cam_obj),
        "z_median": z_median,
        "z_min": float(np.min(verts_cam[:, 2])),
        "z_max": float(np.max(verts_cam[:, 2])),
    }
    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    if coord_meta is None:
        return verts_cam, diag

    cam_t_cfg = coord_meta.get("cam_t", None)
    local_obj_cfg = coord_meta.get("pose_local_obj", None)
    if cam_t_cfg is None or local_obj_cfg is None:
        return verts_cam, diag

    local_obj = _resolve_meta_path(final_dir, local_obj_cfg)
    if not local_obj.exists():
        return verts_cam, diag

    verts_local = _load_obj_vertices(local_obj)
    if verts_local.shape != verts_cam.shape:
        raise RuntimeError(
            "Coordinate contract failed: local/camera OBJ vertex shape mismatch: "
            f"{verts_local.shape} vs {verts_cam.shape}."
        )
    cam_t = np.asarray(cam_t_cfg, dtype=np.float32).reshape(3,)
    delta = verts_cam - verts_local
    delta_median = np.median(delta, axis=0).astype(np.float32)
    max_err = float(np.max(np.abs(delta_median - cam_t)))
    diag.update(
        {
            "pose_local_obj": str(local_obj),
            "cam_t": cam_t.tolist(),
            "delta_median_local_to_camera": delta_median.tolist(),
            "cam_t_contract_max_abs_err": max_err,
        }
    )
    if max_err > float(coord_tol):
        raise RuntimeError(
            "Coordinate contract failed: (camera_world_obj - pose_local_obj) median does not match cam_t. "
            f"max_abs_err={max_err:.6f} > tol={float(coord_tol):.6f}. "
            "Please rerun WHAM+PyMAF export."
        )
    return verts_cam, diag


def _copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    n = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        n += len(list(dir_path.glob(ext)))
    return int(n)


def _build_bind_camera_json(
    *,
    frame_path: Path,
    focal_px: float,
) -> dict:
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read reference frame for camera override: {frame_path}")
    h, w = int(img.shape[0]), int(img.shape[1])
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    f = float(focal_px)
    return {
        "type": "pinhole",
        "K": [
            f, 0.0, cx,
            0.0, f, cy,
            0.0, 0.0, 1.0,
        ],
        "R": [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        "t": [0.0, 0.0, 0.0],
        "image_width": int(w),
        "image_height": int(h),
        "source": "wham_pymaf_bind_override",
    }


def _resolve_render_local_frame_index(stage2_ckpt: Path, reference_frame_name: str) -> tuple[int, int]:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Cannot import torch to read stage2 checkpoint: {e}")

    ckpt = torch.load(str(stage2_ckpt), map_location="cpu")
    frames = ckpt.get("frames", None)
    if not isinstance(frames, (list, tuple)) or len(frames) <= 0:
        raise RuntimeError(
            f"Checkpoint has no valid frame list: {stage2_ckpt}. "
            "Expected a non-empty `frames` entry."
        )

    frame_names = [str(x) for x in frames]
    n_ckpt = int(len(frame_names))
    if n_ckpt == 1:
        # Single-frame checkpoint always maps to local index 0.
        return 0, n_ckpt

    if reference_frame_name in frame_names:
        return int(frame_names.index(reference_frame_name)), n_ckpt

    raise RuntimeError(
        "Reference frame not found in checkpoint frames: "
        f"`{reference_frame_name}` not in {frame_names}. "
        "Please ensure stage2 training and render reference are built from the same frame set."
    )


def _list_frame_paths(frames_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p.resolve()
    return None


def _build_pymaf_smpl_assets(
    *,
    project_root: Path,
    final_dir: Path,
    frame_name: str,
    verts_cam_ref: np.ndarray,
) -> dict:
    out: dict = {
        "smpl_pose_joints_cam": None,
        "smpl_tpose_joints_cam": None,
        "smpl_joint_parents": None,
        "smplx_tpose_verts_cam": None,
        "smplx_tpose_joints_cam": None,
        "sources": {},
    }

    pred_results, rec_idx = _load_pymaf_frame_record(final_dir=final_dir, frame_name=frame_name)
    cam_t = np.asarray(pred_results["orig_cam_t"][rec_idx], dtype=np.float32).reshape(1, 3)

    # ---- SMPL joints (posed + T-pose) from pose(72)/betas(10) ----
    try:
        import torch  # type: ignore
        import smplx  # type: ignore

        pose72 = np.asarray(pred_results["pose"][rec_idx], dtype=np.float32).reshape(1, 72)
        betas10 = np.asarray(pred_results["betas"][rec_idx], dtype=np.float32).reshape(1, 10)

        smpl_model_path = _first_existing(
            [
                project_root / "assets" / "smpl" / "SMPL_N_model.npz",
                project_root / "assets" / "pymafx_data" / "data" / "smpl" / "SMPL_NEUTRAL.npz",
            ]
        )
        if smpl_model_path is not None:
            smpl_model = smplx.SMPL(
                model_path=str(smpl_model_path),
                ext="npz",
                batch_size=1,
            )
            pose_t = torch.from_numpy(pose72)
            betas_t = torch.from_numpy(betas10)
            cam_t_t = torch.from_numpy(cam_t.astype(np.float32))
            out_pose = smpl_model(
                betas=betas_t,
                global_orient=pose_t[:, :3],
                body_pose=pose_t[:, 3:],
                transl=cam_t_t,
            )
            out_tpose = smpl_model(
                betas=betas_t,
                global_orient=torch.zeros((1, 3), dtype=torch.float32),
                body_pose=torch.zeros((1, 69), dtype=torch.float32),
                transl=cam_t_t,
            )
            out["smpl_pose_joints_cam"] = out_pose.joints.detach().cpu().numpy()[0].astype(np.float32)
            out["smpl_tpose_joints_cam"] = out_tpose.joints.detach().cpu().numpy()[0].astype(np.float32)
            out["smpl_joint_parents"] = np.asarray(smpl_model.parents.detach().cpu().numpy(), dtype=np.int32)
            out["sources"]["smpl_joints"] = str(smpl_model_path)
    except Exception as e:
        out["sources"]["smpl_joints_error"] = str(e)

    # Fallback posed joints from PyMAF output
    if out["smpl_pose_joints_cam"] is None and ("joints3d" in pred_results):
        j3d = np.asarray(pred_results["joints3d"][rec_idx], dtype=np.float32)
        out["smpl_pose_joints_cam"] = (j3d + cam_t.astype(np.float32)).astype(np.float32)
        out["sources"]["smpl_joints_fallback"] = "pymaf_joints3d_plus_cam_t"

    # ---- SMPL-X T-pose mesh (same topology as SMPL-X bind mesh) ----
    try:
        import torch  # type: ignore
        import smplx  # type: ignore

        smplx_model_path = _first_existing(
            [
                project_root / "assets" / "smplx" / "SMPLX_NEUTRAL.npz",
                project_root / "assets" / "pymafx_data" / "data" / "smpl" / "SMPLX_NEUTRAL_2020.npz",
                project_root / "assets" / "pymafx_data" / "data" / "smpl" / "SMPLX_NEUTRAL.npz",
            ]
        )
        if smplx_model_path is not None:
            smplx_model = smplx.SMPLX(
                model_path=str(smplx_model_path),
                ext="npz",
                use_pca=False,
                batch_size=1,
            )

            betas_x = None
            smplx_params = pred_results.get("smplx_params", None)
            if isinstance(smplx_params, (list, tuple)) and len(smplx_params) > rec_idx:
                rec = smplx_params[rec_idx]
                if isinstance(rec, dict) and ("shape" in rec):
                    betas_x = np.asarray(rec["shape"], dtype=np.float32).reshape(1, -1)
            if betas_x is None:
                betas_x = np.asarray(pred_results["betas"][rec_idx], dtype=np.float32).reshape(1, -1)
            nb = int(getattr(smplx_model, "num_betas", betas_x.shape[1]))
            betas_x = betas_x[:, :nb]
            if betas_x.shape[1] < nb:
                pad = np.zeros((1, nb - betas_x.shape[1]), dtype=np.float32)
                betas_x = np.concatenate([betas_x, pad], axis=1)

            ne = int(getattr(smplx_model, "num_expression_coeffs", 10))
            expr = np.zeros((1, ne), dtype=np.float32)

            cam_t_t = torch.from_numpy(cam_t.astype(np.float32))
            out_x = smplx_model(
                betas=torch.from_numpy(betas_x.astype(np.float32)),
                global_orient=torch.zeros((1, 3), dtype=torch.float32),
                body_pose=torch.zeros((1, int(getattr(smplx_model, "NUM_BODY_JOINTS", 21)) * 3), dtype=torch.float32),
                left_hand_pose=torch.zeros((1, int(getattr(smplx_model, "NUM_HAND_JOINTS", 15)) * 3), dtype=torch.float32),
                right_hand_pose=torch.zeros((1, int(getattr(smplx_model, "NUM_HAND_JOINTS", 15)) * 3), dtype=torch.float32),
                jaw_pose=torch.zeros((1, 3), dtype=torch.float32),
                leye_pose=torch.zeros((1, 3), dtype=torch.float32),
                reye_pose=torch.zeros((1, 3), dtype=torch.float32),
                expression=torch.from_numpy(expr),
                transl=cam_t_t,
            )
            verts_tpose = out_x.vertices.detach().cpu().numpy()[0].astype(np.float32)
            if int(verts_tpose.shape[0]) == int(verts_cam_ref.shape[0]):
                out["smplx_tpose_verts_cam"] = verts_tpose
            out["smplx_tpose_joints_cam"] = out_x.joints.detach().cpu().numpy()[0].astype(np.float32)
            out["sources"]["smplx_tpose"] = str(smplx_model_path)
    except Exception as e:
        out["sources"]["smplx_tpose_error"] = str(e)

    return out


def _resolve_frame_index(frames_dir: Path, frame_name: str) -> int:
    frames = _list_frame_paths(frames_dir)
    if len(frames) <= 0:
        raise RuntimeError(f"No frames found in {frames_dir}")
    names = [p.name for p in frames]
    if frame_name not in names:
        raise RuntimeError(
            f"Reference frame `{frame_name}` not found in frames_dir: {frames_dir}. "
            "Please make sure WHAM/PyMAF output and current frames_dir match."
        )
    return int(names.index(frame_name))


def main() -> None:
    parser = argparse.ArgumentParser("Build strict bound stage2 assets from WHAM+PyMAF final output and run stage2+render")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--wham_pymaf_final_dir",
        type=str,
        default="wham_pymaf_test/output_full_run/final",
        help="Directory containing last_frame_* outputs from run_wham_pymaf_test.py",
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        default="outputs/stage1",
        help="Existing stage1 dir providing cameras/keypoints/person masks",
    )
    parser.add_argument(
        "--stage1_bind_out_dir",
        type=str,
        default="outputs/stage1_wham_pymaf_bind",
        help="Generated bind asset dir",
    )
    parser.add_argument(
        "--stage2_out_dir",
        type=str,
        default="outputs/stage2_wham_pymaf_bind",
        help="Stage2 output directory",
    )
    parser.add_argument(
        "--renders_out_dir",
        type=str,
        default="outputs/renders_stage2_wham_pymaf_bind_3views",
        help="Render output directory",
    )
    parser.add_argument("--config", type=str, default="configs/stage2_gaussian.yaml", help="Stage2/render config")
    parser.add_argument("--iters", type=int, default=3000, help="Stage2 iterations")
    parser.add_argument("--z_median_min", type=float, default=0.5, help="Sanity threshold for camera-space z median")
    parser.add_argument(
        "--bind_camera_focal",
        type=float,
        default=5000.0,
        help="Camera focal length (px) written to bind cameras.json to match WHAM+PyMAF contour projection.",
    )
    parser.add_argument(
        "--no_bind_camera_override",
        action="store_true",
        help="Do not override bind cameras.json; keep copied stage1 camera file.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--skip_render", action="store_true", help="Skip render stage")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    final_dir = _resolve_path(project_root, args.wham_pymaf_final_dir)
    stage1_dir = _resolve_path(project_root, args.stage1_dir)
    bind_dir = _resolve_path(project_root, args.stage1_bind_out_dir)
    stage2_out = _resolve_path(project_root, args.stage2_out_dir)
    renders_out = _resolve_path(project_root, args.renders_out_dir)
    cfg_path = _resolve_path(project_root, args.config)
    frames_dir = project_root / "data/frames/input"

    if not final_dir.exists():
        raise FileNotFoundError(f"wham_pymaf_final_dir not found: {final_dir}")
    if not stage1_dir.exists():
        raise FileNotFoundError(f"stage1_dir not found: {stage1_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    frame_name = _discover_frame_name(final_dir=final_dir, fallback_frames_dir=frames_dir)
    frame_idx = _resolve_frame_index(frames_dir=frames_dir, frame_name=frame_name)
    cam_obj = _resolve_or_make_cam_obj(final_dir=final_dir, frame_name=frame_name)
    verts_cam, coord_diag = _validate_coordinate_contract(
        final_dir=final_dir,
        cam_obj=cam_obj,
        z_median_min=float(args.z_median_min),
        coord_tol=1e-3,
    )
    z_median = float(coord_diag["z_median"])
    smpl_assets = _build_pymaf_smpl_assets(
        project_root=project_root,
        final_dir=final_dir,
        frame_name=frame_name,
        verts_cam_ref=verts_cam,
    )

    (bind_dir / "smpl_mesh_sequence").mkdir(parents=True, exist_ok=True)
    verts_tpose_cam = smpl_assets.get("smplx_tpose_verts_cam", None)
    if isinstance(verts_tpose_cam, np.ndarray) and verts_tpose_cam.shape == verts_cam.shape:
        _write_obj_replace_vertices(cam_obj, bind_dir / "smpl_mesh_tpose.obj", verts_tpose_cam)
    else:
        shutil.copy2(cam_obj, bind_dir / "smpl_mesh_tpose.obj")
    shutil.copy2(cam_obj, bind_dir / "smpl_mesh_sequence" / "000000.obj")

    np.savez(
        bind_dir / "smpl_vertices_sequence.npz",
        verts_cam=verts_cam[None].astype(np.float32),
        frame_names=np.asarray([frame_name], dtype=object),
    )
    np.savez(
        bind_dir / "smpl_params_sequence.npz",
        frame_names=np.asarray([frame_name], dtype=object),
        fit_strategy=np.asarray(["stage1_motion_prior"], dtype=object),
        export_upright_rot=np.eye(3, dtype=np.float32),
        export_center_xy=np.asarray([0], dtype=np.int32),
        export_center_xy_value=np.zeros((2,), dtype=np.float32),
        export_on_floor=np.asarray([0], dtype=np.int32),
        export_floor_z=np.asarray([0.0], dtype=np.float32),
        verts_tpose_cam=(
            np.asarray(verts_tpose_cam, dtype=np.float32)
            if isinstance(verts_tpose_cam, np.ndarray)
            else np.asarray(verts_cam, dtype=np.float32)
        ),
        joints_tpose_cam=(
            np.asarray(smpl_assets["smplx_tpose_joints_cam"], dtype=np.float32)
            if isinstance(smpl_assets.get("smplx_tpose_joints_cam", None), np.ndarray)
            else (
                np.asarray(smpl_assets["smpl_tpose_joints_cam"], dtype=np.float32)
                if isinstance(smpl_assets.get("smpl_tpose_joints_cam", None), np.ndarray)
                else np.zeros((24, 3), dtype=np.float32)
            )
        ),
    )

    required_stage1_files = [
        "cameras.json",
        "keypoints.json",
        "person_masks.npz",
    ]
    optional_stage1_files: list[str] = []
    copied = {}
    for name in required_stage1_files + optional_stage1_files:
        copied[name] = _copy_optional(stage1_dir / name, bind_dir / name)
    for name in required_stage1_files:
        if not bool(copied.get(name, False)):
            raise FileNotFoundError(f"Required file missing: {stage1_dir / name}")

    pose_joints_cam = smpl_assets.get("smpl_pose_joints_cam", None)
    joint_parents = smpl_assets.get("smpl_joint_parents", None)
    if isinstance(pose_joints_cam, np.ndarray) and pose_joints_cam.ndim == 2 and pose_joints_cam.shape[1] == 3:
        pose_joints_cam = pose_joints_cam.astype(np.float32)
        skel_npz = {
            "frame_names": np.asarray([frame_name], dtype=object),
            "joints_cam_theta": pose_joints_cam[None],
            "model_type": np.asarray(["smpl"], dtype=object),
            "gender": np.asarray(["neutral"], dtype=object),
            "source": np.asarray(["wham_pymaf_bind"], dtype=object),
        }
        joints_npz = {
            "frame_names": np.asarray([frame_name], dtype=object),
            "joints_cam": pose_joints_cam[None],
            "model_type": np.asarray(["smpl"], dtype=object),
            "gender": np.asarray(["neutral"], dtype=object),
            "source": np.asarray(["wham_pymaf_bind"], dtype=object),
        }
        if isinstance(joint_parents, np.ndarray) and joint_parents.ndim == 1:
            skel_npz["joint_parents"] = joint_parents.astype(np.int32)
            joints_npz["joint_parents"] = joint_parents.astype(np.int32)
        np.savez(bind_dir / "smpl_skeleton_sequence.npz", **skel_npz)
        np.savez(bind_dir / "smpl_joints_sequence.npz", **joints_npz)
        copied["smpl_skeleton_sequence.npz"] = True
        copied["smpl_joints_sequence.npz"] = True
    else:
        copied["smpl_skeleton_sequence.npz"] = False
        copied["smpl_joints_sequence.npz"] = False

    bind_camera_info = {"override_applied": False}
    if not bool(args.no_bind_camera_override):
        ref_frame_path = frames_dir / frame_name
        bind_cam = _build_bind_camera_json(
            frame_path=ref_frame_path,
            focal_px=float(args.bind_camera_focal),
        )
        with (bind_dir / "cameras.json").open("w", encoding="utf-8") as f:
            json.dump(bind_cam, f, indent=2)
        bind_camera_info = {
            "override_applied": True,
            "camera_json_path": str(bind_dir / "cameras.json"),
            "focal_px": float(args.bind_camera_focal),
            "reference_frame": str(ref_frame_path),
            "K": bind_cam["K"],
            "R": bind_cam["R"],
            "t": bind_cam["t"],
        }

    coord_meta = _load_json_if_exists(final_dir / "last_frame_coord_meta.json")
    t_edit2world = np.eye(4, dtype=np.float32)
    if coord_meta is not None and coord_meta.get("cam_t", None) is not None:
        cam_t = np.asarray(coord_meta["cam_t"], dtype=np.float32).reshape(3,)
        t_edit2world[:3, 3] = cam_t
    t_world2edit = np.linalg.inv(t_edit2world).astype(np.float32)

    bind_report = {
        "wham_pymaf_final_dir": str(final_dir),
        "stage1_source_dir": str(stage1_dir),
        "bind_dir": str(bind_dir),
        "camera_world_obj": str(cam_obj),
        "frame_name": frame_name,
        "frame_index": int(frame_idx),
        "verts_cam_shape": list(verts_cam.shape),
        "coord_diag": coord_diag,
        "copied_stage1_files": copied,
        "bind_camera": bind_camera_info,
        "anchor_spaces": {
            "training_render_space": "camera_world_metric",
            "edit_space": "smpl_pose_local",
            "T_edit2world": t_edit2world.tolist(),
            "T_world2edit": t_world2edit.tolist(),
        },
    }
    with (bind_dir / "bind_asset_report.json").open("w", encoding="utf-8") as f:
        json.dump(bind_report, f, indent=2)

    env = os.environ.copy()
    py_path = str(project_root / "src")
    env["PYTHONPATH"] = py_path if "PYTHONPATH" not in env else f"{py_path}{os.pathsep}{env['PYTHONPATH']}"

    stage2_cmd = [
        str(args.python),
        "-m",
        "avatar_gs.cli.stage2_train_gaussian",
        "--config",
        str(cfg_path),
        "--override",
        f"stage2.stage1_dir={bind_dir}",
        f"stage2.init_mesh_obj={bind_dir / 'smpl_mesh_sequence' / '000000.obj'}",
        f"stage2.out_dir={stage2_out}",
        "stage2.frame_select=indices",
        f"stage2.frame_indices={int(frame_idx)}",
        "stage2.n_frames=1",
        "stage2.bind_to_mesh=true",
        f"stage2.bind_mesh_sequence_dir={bind_dir / 'smpl_mesh_sequence'}",
        f"stage2.bind_sequence_npz={bind_dir / 'smpl_params_sequence.npz'}",
        f"stage2.bind_vertices_npz={bind_dir / 'smpl_vertices_sequence.npz'}",
        f"stage2.bind_tpose_obj={bind_dir / 'smpl_mesh_tpose.obj'}",
        "stage2.bind_mesh_to_camera=true",
        "stage2.bind_mesh_obj_space=camera",
        "stage2.bind_inverse_export_upright=false",
        f"stage2.keypoints_json={bind_dir / 'keypoints.json'}",
        f"stage2.person_masks_npz={bind_dir / 'person_masks.npz'}",
        f"stage2.bind_rigid_align.person_masks_npz={bind_dir / 'person_masks.npz'}",
        f"stage2.iters={int(args.iters)}",
    ]
    _run_cmd(stage2_cmd, cwd=project_root, env=env)
    stage2_ckpt = stage2_out / "gaussians3d.pt"
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found after training: {stage2_ckpt}")
    render_local_idx, ckpt_frames_count = _resolve_render_local_frame_index(
        stage2_ckpt=stage2_ckpt,
        reference_frame_name=frame_name,
    )

    chain_summary = {
        "bind_dir": str(bind_dir),
        "stage2_out_dir": str(stage2_out),
        "stage2_ckpt": str(stage2_ckpt),
        "renders_out_dir": str(renders_out),
        "coord_diag": coord_diag,
        "frame_name": frame_name,
        "frame_index": int(frame_idx),
        "reference_frame_name": frame_name,
        "stage2_global_frame_index": int(frame_idx),
        "render_local_frame_index": int(render_local_idx),
        "ckpt_frames_count": int(ckpt_frames_count),
        "bind_camera": bind_camera_info,
        "smpl_asset_sources": smpl_assets.get("sources", {}),
        "has_smpl_pose_joints_cam": bool(isinstance(smpl_assets.get("smpl_pose_joints_cam", None), np.ndarray)),
        "has_smplx_tpose_verts_cam": bool(isinstance(smpl_assets.get("smplx_tpose_verts_cam", None), np.ndarray)),
    }

    if not bool(args.skip_render):
        render_cmd = [
            str(args.python),
            "-m",
            "avatar_gs.cli.render",
            "--config",
            str(cfg_path),
            "--override",
            f"render.stage2_dir={stage2_out}",
            f"render.out_dir={renders_out}",
            "render.frame_select=indices",
            f"render.frame_indices={int(render_local_idx)}",
            "render.max_frames=1",
            "render.real_out_subdir=real",
            "render.fg_nobg_out_subdir=fg_only",
            "render.tpose_out_subdir=t-pose",
            "render.bind_to_mesh=true",
            "render.strict_binding_topology=true",
            f"render.bind_mesh_sequence_dir={bind_dir / 'smpl_mesh_sequence'}",
            f"render.bind_sequence_npz={bind_dir / 'smpl_params_sequence.npz'}",
            f"render.bind_vertices_npz={bind_dir / 'smpl_vertices_sequence.npz'}",
            f"render.tpose_mesh_obj={bind_dir / 'smpl_mesh_tpose.obj'}",
            "render.bind_mesh_to_camera=true",
            "render.bind_mesh_obj_space=camera",
            "render.tpose_mesh_obj_space=camera",
            "render.bind_inverse_export_upright=false",
            "render.save_main_image=false",
            "render.save_real_image=true",
            "render.save_fg_nobg_skeleton=true",
            "render.render_tpose_image=true",
        ]
        skeleton_npz = bind_dir / "smpl_skeleton_sequence.npz"
        joints_npz = bind_dir / "smpl_joints_sequence.npz"
        if skeleton_npz.exists():
            render_cmd.append(f"render.smpl_skeleton_npz={skeleton_npz}")
        if joints_npz.exists():
            render_cmd.append(f"render.smpl_joints_npz={joints_npz}")
        if (not skeleton_npz.exists()) and (not joints_npz.exists()):
            render_cmd.append("render.overlay_smpl_skeleton=false")
        _run_cmd(render_cmd, cwd=project_root, env=env)

        real_dir = renders_out / "real"
        fg_only_dir = renders_out / "fg_only"
        tpose_dir = renders_out / "t-pose"
        real_n = _count_images(real_dir)
        fg_n = _count_images(fg_only_dir)
        tp_n = _count_images(tpose_dir)
        if real_n <= 0 or fg_n <= 0 or tp_n <= 0:
            raise RuntimeError(
                "Render outputs are incomplete: "
                f"real={real_n}, fg_only={fg_n}, t-pose={tp_n}. "
                "Please inspect render logs."
            )
        chain_summary["render_outputs"] = {
            "real_dir": str(real_dir),
            "fg_only_dir": str(fg_only_dir),
            "tpose_dir": str(tpose_dir),
            "real_count": int(real_n),
            "fg_only_count": int(fg_n),
            "tpose_count": int(tp_n),
        }

    chain_summary_path = bind_dir / "stage2_chain_summary.json"
    with chain_summary_path.open("w", encoding="utf-8") as f:
        json.dump(chain_summary, f, indent=2)

    print("Done.")
    print(f"Bind assets: {bind_dir}")
    print(f"Stage2 output: {stage2_out}")
    print(f"Stage2 ckpt: {stage2_ckpt}")
    print(f"Chain summary: {chain_summary_path}")
    if not bool(args.skip_render):
        print(f"Renders output: {renders_out}")


if __name__ == "__main__":
    main()
