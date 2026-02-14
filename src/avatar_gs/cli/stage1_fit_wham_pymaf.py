from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np


@dataclass
class MeshFrameResult:
    frame_name: str
    frame_path: Path
    verts: np.ndarray
    faces: np.ndarray
    cam_t: np.ndarray
    model_type: str


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _find_first_file(root: Path, name: str) -> Path | None:
    cands = list(root.rglob(name))
    return cands[0] if len(cands) > 0 else None


def _find_any_file(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        cands = list(root.rglob(pat))
        if len(cands) > 0:
            return cands[0]
    return None


def _ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def _find_first_file_in_roots(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        if root is None or (not root.exists()):
            continue
        src = _find_first_file(root, name)
        if src is not None:
            return src
    return None


def _try_download_wham_body_models(runtime_root: Path) -> Path | None:
    """
    Download WHAM body_models bundle that contains auxiliary regressors/faces/mean params.
    Returns extraction directory on success, otherwise None.
    """
    url = "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t"
    dl_dir = runtime_root / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dl_dir / "wham_body_models.tar.gz"
    extract_dir = dl_dir / "wham_body_models_extracted"
    try:
        urllib.request.urlretrieve(url, str(tar_path))
    except Exception:
        return None
    if not tar_path.exists():
        return None
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=str(extract_dir))
    except Exception:
        return None
    return extract_dir


def _try_write_smpl_faces_npy(body_models_dir: Path, search_roots: list[Path]) -> bool:
    """Generate smpl_faces.npy from neutral SMPL npz when missing."""
    dst = body_models_dir / "smpl_faces.npy"
    if dst.exists():
        return True

    neutral_npz = None
    for cand_name in ["SMPL_N_model.npz", "SMPL_NEUTRAL.npz"]:
        neutral_npz = _find_first_file_in_roots(search_roots, cand_name)
        if neutral_npz is not None:
            break
    if neutral_npz is None:
        return False

    try:
        data = np.load(str(neutral_npz), allow_pickle=True)
        faces = None
        for key in ["f", "faces", "F"]:
            if key in data.files:
                faces = np.asarray(data[key], dtype=np.int64)
                break
        if faces is None:
            return False
        np.save(str(dst), faces)
        return True
    except Exception:
        return False


def _prepare_runtime_wham_assets(
    assets_dir: Path,
    runtime_root: Path,
    wham_repo_dir: Path | None = None,
    pymaf_repo_dir: Path | None = None,
) -> Path:
    wham_root = runtime_root / "wham_runtime_assets"
    body_models_dir = wham_root / "body_models"
    smpl_dir = body_models_dir / "smpl"
    ckpt_dir = wham_root / "checkpoints"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # WHAM uses smplx.SMPL, which expects pkl model files under body_models/smpl/.
    smpl_pkl_maps = [
        (["SMPL_NEUTRAL.pkl", "SMPL_N_model.pkl"], "SMPL_NEUTRAL.pkl"),
        (["SMPL_MALE.pkl", "SMPL_M_model.pkl"], "SMPL_MALE.pkl"),
        (["SMPL_FEMALE.pkl", "SMPL_F_model.pkl"], "SMPL_FEMALE.pkl"),
    ]
    missing_smpl_pkls: list[str] = []
    for src_names, dst_name in smpl_pkl_maps:
        src = None
        for src_name in src_names:
            cand = assets_dir / "smpl" / src_name
            if cand.exists():
                src = cand
                break
        if src is None:
            missing_smpl_pkls.append(dst_name)
            continue
        _ensure_link_or_copy(src, smpl_dir / dst_name)

    # Keep legacy npz links if present (used by some auxiliary tools).
    legacy_npz_maps = [
        ("SMPL_N_model.npz", "SMPL_NEUTRAL.npz"),
        ("SMPL_M_model.npz", "SMPL_MALE.npz"),
        ("SMPL_F_model.npz", "SMPL_FEMALE.npz"),
    ]
    for src_name, dst_name in legacy_npz_maps:
        src = assets_dir / "smpl" / src_name
        if src.exists():
            _ensure_link_or_copy(src, smpl_dir / dst_name)

    if len(missing_smpl_pkls) > 0:
        raise FileNotFoundError(
            "WHAM requires SMPL pkl files under assets/smpl. Missing: "
            + ", ".join(missing_smpl_pkls)
        )

    aux_files = [
        "J_regressor_wham.npy",
        "J_regressor_h36m.npy",
        "J_regressor_feet.npy",
        "J_regressor_extra.npy",
        "smpl_mean_params.npz",
        "smpl_faces.npy",
        "smplx2smpl.pkl",
    ]
    search_roots: list[Path] = [assets_dir]
    if pymaf_repo_dir is not None:
        search_roots.extend([pymaf_repo_dir / "data", pymaf_repo_dir])
    if wham_repo_dir is not None:
        search_roots.extend([wham_repo_dir / "dataset" / "body_models", wham_repo_dir])

    missing_aux: list[str] = []
    for name in aux_files:
        dst = body_models_dir / name
        if dst.exists():
            continue
        src = _find_first_file_in_roots(search_roots, name)
        if src is not None:
            _ensure_link_or_copy(src, dst)
        else:
            missing_aux.append(name)

    if len(missing_aux) > 0:
        extracted_root = _try_download_wham_body_models(runtime_root=runtime_root)
        if extracted_root is not None:
            search_roots = [extracted_root] + search_roots
            missing_aux_2: list[str] = []
            for name in aux_files:
                dst = body_models_dir / name
                if dst.exists():
                    continue
                src = _find_first_file_in_roots(search_roots, name)
                if src is not None:
                    _ensure_link_or_copy(src, dst)
                else:
                    missing_aux_2.append(name)
            missing_aux = missing_aux_2

    if "smpl_faces.npy" in missing_aux:
        if _try_write_smpl_faces_npy(body_models_dir=body_models_dir, search_roots=search_roots):
            missing_aux = [x for x in missing_aux if x != "smpl_faces.npy"]

    if len(missing_aux) > 0:
        raise FileNotFoundError(
            "Missing WHAM body-model auxiliary files: "
            + ", ".join(missing_aux)
            + ". Put them under assets/ (or third_party/WHAM dataset body_models), "
            + "or allow auto-download of WHAM body_models bundle."
        )

    for name in [
        "wham_vit_bedlam_w_3dpw.pth.tar",
        "vitpose-h-multi-coco.pth",
        "yolov8x.pt",
        "hmr2a.ckpt",
    ]:
        src = _find_first_file(assets_dir, name)
        if src is not None:
            _ensure_link_or_copy(src, ckpt_dir / name)

    return wham_root


def _resolve_wham_ckpts(assets_dir: Path, wham_runtime_root: Path) -> dict[str, Path | None]:
    ckpt_dir = wham_runtime_root / "checkpoints"
    out: dict[str, Path | None] = {}
    out["wham"] = _find_any_file(ckpt_dir, ["wham_vit_bedlam_w_3dpw.pth.tar", "wham_vit*.pth.tar", "*.pth.tar"])
    out["vitpose"] = _find_any_file(ckpt_dir, ["vitpose-h-multi-coco.pth"])
    out["yolo"] = _find_any_file(ckpt_dir, ["yolov8x.pt"])
    out["hmr2a"] = _find_any_file(ckpt_dir, ["hmr2a.ckpt"])

    if out["wham"] is None:
        out["wham"] = _find_any_file(assets_dir, ["wham_vit_bedlam_w_3dpw.pth.tar", "wham_vit*.pth.tar", "*.pth.tar"])
    if out["vitpose"] is None:
        out["vitpose"] = _find_any_file(assets_dir, ["vitpose-h-multi-coco.pth"])
    if out["yolo"] is None:
        out["yolo"] = _find_any_file(assets_dir, ["yolov8x.pt"])
    if out["hmr2a"] is None:
        out["hmr2a"] = _find_any_file(assets_dir, ["hmr2a.ckpt"])
    return out


def _apply_wham_ckpt_overrides(
    wham_ckpts: dict[str, Path | None],
    wham_runtime_root: Path,
    wham_ckpt: Path | None,
    vitpose_ckpt: Path | None,
    yolo_ckpt: Path | None,
    hmr2_ckpt: Path | None,
) -> dict[str, Path | None]:
    ckpt_dir = wham_runtime_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = dict(wham_ckpts)

    mapping = [
        ("wham", wham_ckpt, "wham_vit_bedlam_w_3dpw.pth.tar"),
        ("vitpose", vitpose_ckpt, "vitpose-h-multi-coco.pth"),
        ("yolo", yolo_ckpt, "yolov8x.pt"),
        ("hmr2a", hmr2_ckpt, "hmr2a.ckpt"),
    ]
    for key, src, dst_name in mapping:
        if src is None:
            continue
        if not src.exists():
            raise FileNotFoundError(f"Override checkpoint not found: {src}")
        dst = ckpt_dir / dst_name
        _ensure_link_or_copy(src, dst)
        out[key] = dst
    return out


def _resolve_pymaf_ckpt(assets_dir: Path, override: Path | None) -> Path:
    if override is not None and override.exists():
        return override

    p = _find_any_file(
        assets_dir,
        [
            "PyMAF-X_model_checkpoint_v1.1.pt",
            "PyMAF-X_model_checkpoint_v1.1.pth",
            "PyMAF-X_model_checkpoint_v1.1.pth.tar",
            "PyMAF*.pt",
            "PyMAF*.pth",
            "PyMAF*.pth.tar",
            "*pymaf*.pt",
            "*pymaf*.pth",
            "*pymaf*.pth.tar",
            "*.pt",
            "*.pth",
            "*.pth.tar",
            "*.ckpt",
        ],
    )
    if p is None:
        raise FileNotFoundError("Cannot find PyMAF checkpoint under assets. Please pass --pymaf_ckpt explicitly.")
    return p


def _extract_tar_if_needed(tar_path: Path, dst_dir: Path, marker: str) -> None:
    if (dst_dir / marker).exists():
        return
    if not tar_path.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = str(member.name).replace('\\', '/')
            if name.startswith('data/'):
                name = name[len('data/'):]
            if name == '' or name.endswith('/'):
                continue

            out_path = dst_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with out_path.open('wb') as f:
                f.write(src.read())


def _prepare_pymaf_runtime_data(assets_dir: Path, pymaf_dir: Path, pymaf_ckpt: Path, wham_asset_root: Path | None = None) -> tuple[Path, Path]:
    data_dir = pymaf_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = assets_dir / "pymafx_data" / "data.tar.gz"
    _extract_tar_if_needed(tar_path, data_dir, marker="smpl_mean_params.npz")

    smpl_down_src = _find_any_file(assets_dir, ["smpl_downsampling.npz", "mesh_downsampling.npz"])
    mano_down_src = _find_any_file(assets_dir, ["mano_downsampling.npz"])
    if smpl_down_src is not None:
        _ensure_link_or_copy(smpl_down_src, data_dir / "smpl_downsampling.npz")
    if mano_down_src is not None:
        _ensure_link_or_copy(mano_down_src, data_dir / "mano_downsampling.npz")

    smpl_dir = data_dir / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    maps = [
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.npz", smpl_dir / "SMPLX_NEUTRAL.npz"),
        (assets_dir / "smplx" / "SMPLX_NEUTRAL.npz", smpl_dir / "SMPLX_NEUTRAL_2020.npz"),
        (assets_dir / "smplx" / "SMPLX_MALE.npz", smpl_dir / "SMPLX_MALE.npz"),
        (assets_dir / "smplx" / "SMPLX_FEMALE.npz", smpl_dir / "SMPLX_FEMALE.npz"),
        (assets_dir / "smpl" / "SMPL_N_model.npz", smpl_dir / "SMPL_NEUTRAL.npz"),
        (assets_dir / "smpl" / "SMPL_M_model.npz", smpl_dir / "SMPL_MALE.npz"),
        (assets_dir / "smpl" / "SMPL_F_model.npz", smpl_dir / "SMPL_FEMALE.npz"),
    ]
    for src, dst in maps:
        if src.exists():
            _ensure_link_or_copy(src, dst)

    # Ensure checkpoint sits at PyMAF conventional location too.
    pretrain_dir = data_dir / "pretrained_model"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_copy(pymaf_ckpt, pretrain_dir / pymaf_ckpt.name)

    # PyMAF-X expects data/smpl/model_transfer/smplx_to_smpl.pkl
    transfer_dir = smpl_dir / "model_transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    transfer_dst = transfer_dir / "smplx_to_smpl.pkl"
    if not transfer_dst.exists():
        transfer_src = _find_any_file(assets_dir, ["smplx_to_smpl.pkl", "smplx2smpl.pkl"])
        if transfer_src is None and wham_asset_root is not None:
            transfer_src = _find_any_file(wham_asset_root, ["smplx_to_smpl.pkl", "smplx2smpl.pkl"])
        if transfer_src is not None:
            _ensure_link_or_copy(transfer_src, transfer_dst)

    return data_dir, smpl_dir


def _estimate_person_mask_grabcut(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    rect = (int(0.08 * w), int(0.04 * h), int(0.84 * w), int(0.92 * h))
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image_bgr, gc_mask, rect, bg_model, fg_model, 5, mode=cv2.GC_INIT_WITH_RECT)
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


def _project_verts_to_uv(verts: np.ndarray, cam_t: np.ndarray, width: int, height: int, focal: float = 5000.0) -> tuple[np.ndarray, np.ndarray]:
    v_cam = verts + cam_t.reshape(1, 3)
    z = np.clip(v_cam[:, 2], 1e-6, None)
    u = float(focal) * (v_cam[:, 0] / z) + 0.5 * float(width)
    v = float(focal) * (v_cam[:, 1] / z) + 0.5 * float(height)
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & (v_cam[:, 2] > 1e-5)
    return uv, valid


def _rasterize_mesh_mask(uv: np.ndarray, valid_v: np.ndarray, faces: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a >= len(valid_v) or b >= len(valid_v) or c >= len(valid_v):
            continue
        if (not valid_v[a]) or (not valid_v[b]) or (not valid_v[c]):
            continue
        pts = np.round(uv[[a, b, c], :]).astype(np.int32)
        xmin = int(np.min(pts[:, 0]))
        xmax = int(np.max(pts[:, 0]))
        ymin = int(np.min(pts[:, 1]))
        ymax = int(np.max(pts[:, 1]))
        if xmax < 0 or ymax < 0 or xmin >= w or ymin >= h:
            continue
        cv2.fillConvexPoly(mask, pts, color=255, lineType=cv2.LINE_8)
    return mask


def _largest_contour(mask_u8: np.ndarray) -> list[np.ndarray]:
    cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[:1]


def _save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {float(v[0]):.7f} {float(v[1]):.7f} {float(v[2]):.7f}\\n")
        f1 = faces.astype(np.int64) + 1
        for tri in f1:
            f.write(f"f {int(tri[0])} {int(tri[1])} {int(tri[2])}\\n")


def _load_faces_from_npz(npz_path: Path) -> np.ndarray:
    arr = np.load(str(npz_path), allow_pickle=True)
    if "f" in arr.files:
        return np.asarray(arr["f"], dtype=np.int32)
    if "faces" in arr.files:
        return np.asarray(arr["faces"], dtype=np.int32)
    raise KeyError(f"No face topology key found in {npz_path} (expected 'f' or 'faces').")


def _select_last_frame_result(
    pred_results: dict,
    selected_meta: dict,
    gt_last_frame_name: str,
    faces_smplx: np.ndarray | None,
    faces_smpl: np.ndarray | None,
) -> MeshFrameResult:
    sel_files = [str(x) for x in selected_meta.get("selected_files", [])]
    if len(sel_files) == 0:
        raise RuntimeError("selected_frames_meta.json has empty selected_files.")

    if gt_last_frame_name in sel_files:
        local_idx = int(sel_files.index(gt_last_frame_name))
        frame_name = gt_last_frame_name
    else:
        local_idx = int(len(sel_files) - 1)
        frame_name = sel_files[local_idx]

    frame_ids = np.asarray(pred_results["frame_ids"], dtype=np.int64).reshape(-1)
    cand = np.where(frame_ids == local_idx)[0]
    if cand.size <= 0:
        idx = int(np.argmax(frame_ids))
    else:
        idx = int(cand[0])

    cam_t = np.asarray(pred_results["orig_cam_t"][idx], dtype=np.float32).reshape(3)

    if "smplx_verts" in pred_results and pred_results["smplx_verts"] is not None:
        verts = np.asarray(pred_results["smplx_verts"][idx], dtype=np.float32)
        if faces_smplx is None:
            raise RuntimeError("smplx_verts available but SMPL-X faces are missing.")
        faces = faces_smplx
        model_type = "smplx"
    else:
        verts = np.asarray(pred_results["verts"][idx], dtype=np.float32)
        if faces_smpl is None:
            raise RuntimeError("verts available but SMPL faces are missing.")
        faces = faces_smpl
        model_type = "smpl"

    return MeshFrameResult(
        frame_name=frame_name,
        frame_path=Path(frame_name),
        verts=verts,
        faces=faces,
        cam_t=cam_t,
        model_type=model_type,
    )


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _check_stage1_requirements(assets_dir: Path, wham_dir: Path, pymaf_dir: Path) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"wham": [], "pymaf": [], "python": []}

    # File checks.
    for name in [
        "wham_vit_bedlam_w_3dpw.pth.tar",
        "vitpose-h-multi-coco.pth",
        "yolov8x.pt",
        "hmr2a.ckpt",
    ]:
        if _find_first_file(assets_dir, name) is None:
            missing["wham"].append(name)

    # Required by WHAM body model loader (smplx.SMPL)
    for name in ["SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"]:
        if _find_first_file(assets_dir, name) is None:
            missing["wham"].append(name)

    if _find_any_file(assets_dir, ["PyMAF-X_model_checkpoint_v1.1.pt", "PyMAF*.pt", "*pymaf*.pt"]) is None:
        missing["pymaf"].append("PyMAF-X_model_checkpoint_v1.1.pt")

    if not (wham_dir / "third-party" / "ViTPose").exists():
        missing["wham"].append("third-party/ViTPose (WHAM submodule)")

    # Python package checks (best-effort).
    checks = [
        "loguru",
        "ultralytics",
        "mmpose",
        "mmcv",
        "openpifpaf",
        "smplx",
        "joblib",
    ]
    for mod in checks:
        try:
            __import__(mod)
        except Exception:
            missing["python"].append(mod)

    return missing


def _move_path_to_trash(path: Path, project_root: Path, tag: str) -> Path | None:
    if not path.exists():
        return None
    trash_root = project_root / 'trash' / f'stage1_pruned_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{tag}'
    trash_root.mkdir(parents=True, exist_ok=True)
    dst = trash_root / path.name
    try:
        shutil.move(str(path), str(dst))
        return dst
    except Exception:
        return None




def main() -> None:
    parser = argparse.ArgumentParser("Stage1: WHAM initialization + PyMAF-X refine")
    parser.add_argument("--project_root", type=str, default=".", help="Project root")
    parser.add_argument("--frames_dir", type=str, default="data/frames/input", help="Video frame directory")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Assets root directory")
    parser.add_argument("--out_dir", type=str, default="outputs/stage1_wham_pymaf", help="All outputs are written here")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for temporary WHAM video")
    parser.add_argument("--pymaf_frame_indices", type=str, default="-1", help="Frame indices passed to PyMAF refine")
    parser.add_argument("--pymaf_ckpt", type=str, default="", help="Optional explicit PyMAF checkpoint path")
    parser.add_argument("--wham_ckpt", type=str, default="", help="Optional explicit WHAM main checkpoint path")
    parser.add_argument("--vitpose_ckpt", type=str, default="", help="Optional explicit ViTPose checkpoint path")
    parser.add_argument("--yolo_ckpt", type=str, default="", help="Optional explicit YOLO checkpoint path")
    parser.add_argument("--hmr2_ckpt", type=str, default="", help="Optional explicit HMR2 checkpoint path")
    parser.add_argument("--single_person", action="store_true", help="Set PyMAF single-person mode")
    parser.add_argument("--skip_wham", action="store_true", help="Skip WHAM stage")
    parser.add_argument("--skip_pymaf", action="store_true", help="Skip PyMAF stage")
    parser.add_argument("--wham_repo_dir", type=str, default="third_party/WHAM", help="WHAM repo dir")
    parser.add_argument("--pymaf_repo_dir", type=str, default="third_party/PyMAF-X", help="PyMAF-X repo dir")
    parser.add_argument("--minimal_stage1_outputs", action="store_true", help="Keep only t-pose obj and selected-frame sequence obj; move intermediates to trash")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    assets_dir = (project_root / args.assets_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    wham_dir = (project_root / args.wham_repo_dir).resolve()
    pymaf_dir = (project_root / args.pymaf_repo_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets dir not found: {assets_dir}")
    if not wham_dir.exists():
        raise FileNotFoundError(f"WHAM repo not found: {wham_dir}")
    if not pymaf_dir.exists():
        raise FileNotFoundError(f"PyMAF repo not found: {pymaf_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wham_out = out_dir / "wham"
    pymaf_out = out_dir / "pymaf"
    final_out = out_dir / "final"
    runtime_out = out_dir / "runtime"
    wham_out.mkdir(parents=True, exist_ok=True)
    pymaf_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)
    runtime_out.mkdir(parents=True, exist_ok=True)

    frames = _list_frames(frames_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    gt_last = frames[-1]
    print(f"Frames: {len(frames)}, last frame: {gt_last.name}")

    req = _check_stage1_requirements(assets_dir=assets_dir, wham_dir=wham_dir, pymaf_dir=pymaf_dir)

    wham_asset_root = _prepare_runtime_wham_assets(
        assets_dir=assets_dir,
        runtime_root=runtime_out,
        wham_repo_dir=wham_dir,
        pymaf_repo_dir=pymaf_dir,
    )
    wham_ckpts = _resolve_wham_ckpts(assets_dir=assets_dir, wham_runtime_root=wham_asset_root)
    wham_ckpts = _apply_wham_ckpt_overrides(
        wham_ckpts=wham_ckpts,
        wham_runtime_root=wham_asset_root,
        wham_ckpt=(Path(args.wham_ckpt).resolve() if str(args.wham_ckpt).strip() else None),
        vitpose_ckpt=(Path(args.vitpose_ckpt).resolve() if str(args.vitpose_ckpt).strip() else None),
        yolo_ckpt=(Path(args.yolo_ckpt).resolve() if str(args.yolo_ckpt).strip() else None),
        hmr2_ckpt=(Path(args.hmr2_ckpt).resolve() if str(args.hmr2_ckpt).strip() else None),
    )

    pymaf_ckpt: Path | None = None
    if not bool(args.skip_pymaf):
        pymaf_ckpt = _resolve_pymaf_ckpt(
            assets_dir=assets_dir,
            override=(Path(args.pymaf_ckpt).resolve() if str(args.pymaf_ckpt).strip() else None),
        )
        _prepare_pymaf_runtime_data(assets_dir=assets_dir, pymaf_dir=pymaf_dir, pymaf_ckpt=pymaf_ckpt, wham_asset_root=wham_asset_root)

    env_base = os.environ.copy()
    env_base["WHAM_ASSET_ROOT"] = str(wham_asset_root)
    env_base["WHAM_CHECKPOINT_DIR"] = str(wham_asset_root / "checkpoints")
    if wham_ckpts["wham"] is not None:
        env_base["WHAM_MAIN_CHECKPOINT"] = str(wham_ckpts["wham"])

    missing_wham = [k for k, v in wham_ckpts.items() if v is None]
    if not bool(args.skip_wham) and len(missing_wham) > 0:
        raise FileNotFoundError(
            "WHAM required checkpoints are missing in assets: "
            + ", ".join(missing_wham)
            + ". Expected: wham_vit_bedlam_w_3dpw.pth.tar, vitpose-h-multi-coco.pth, yolov8x.pt, hmr2a.ckpt"
        )

    if not bool(args.skip_wham):
        wham_cmd = [
            sys.executable,
            "-m",
            "avatar_gs.cli._run_wham_on_frames",
            "--frames_dir",
            str(frames_dir),
            "--output_dir",
            str(wham_out),
            "--wham_repo_dir",
            str(wham_dir),
            "--fps",
            str(float(args.fps)),
            "--estimate_local_only",
        ]
        _run_cmd(wham_cmd, cwd=project_root, env=env_base)

    if not bool(args.skip_pymaf):
        pymaf_cmd = [
            sys.executable,
            "-m",
            "avatar_gs.cli._run_pymaf_refine_frames",
            "--frames_dir",
            str(frames_dir),
            "--output_folder",
            str(pymaf_out),
            "--pymaf_repo_dir",
            str(pymaf_dir),
            "--pretrained_model",
            str(pymaf_ckpt),
            "--frame_indices",
            str(args.pymaf_frame_indices),
        ]
        if bool(args.single_person):
            pymaf_cmd.append("--single_person")
        _run_cmd(pymaf_cmd, cwd=project_root, env=env_base)
    else:
        print(f"Skipped PyMAF stage. WHAM outputs are available at: {wham_out}")
        return

    pymaf_result_dir = pymaf_out / "selected_frames"
    pred_pkl = pymaf_result_dir / "output.pkl"
    meta_json = pymaf_result_dir / "selected_frames_meta.json"
    if not pred_pkl.exists():
        raise FileNotFoundError(f"PyMAF output missing: {pred_pkl}")
    if not meta_json.exists():
        raise FileNotFoundError(f"PyMAF selected-frame metadata missing: {meta_json}")

    pred_results = joblib.load(pred_pkl)
    with meta_json.open("r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    faces_smplx = None
    faces_smpl = None
    smplx_face_path = _find_any_file(assets_dir, ["SMPLX_NEUTRAL_2020.npz", "SMPLX_NEUTRAL.npz"])
    if smplx_face_path is not None:
        faces_smplx = _load_faces_from_npz(smplx_face_path)

    smpl_face_path = _find_any_file(assets_dir, ["SMPL_NEUTRAL.npz", "SMPL_N_model.npz"])
    if smpl_face_path is not None:
        faces_smpl = _load_faces_from_npz(smpl_face_path)

    mesh_res = _select_last_frame_result(
        pred_results=pred_results,
        selected_meta=selected_meta,
        gt_last_frame_name=gt_last.name,
        faces_smplx=faces_smplx,
        faces_smpl=faces_smpl,
    )
    mesh_res.frame_path = frames_dir / mesh_res.frame_name
    if not mesh_res.frame_path.exists():
        raise FileNotFoundError(f"Resolved last frame image not found: {mesh_res.frame_path}")

    obj_local_path = final_out / "last_frame_pose_local.obj"
    obj_cam_path = final_out / f"last_frame_{mesh_res.model_type}_cam.obj"
    obj_legacy_path = final_out / f"last_frame_{mesh_res.model_type}.obj"

    verts_local = np.asarray(mesh_res.verts, dtype=np.float32)
    verts_cam = np.asarray(mesh_res.verts + mesh_res.cam_t.reshape(1, 3), dtype=np.float32)

    _save_obj(obj_local_path, verts_local, mesh_res.faces)
    _save_obj(obj_cam_path, verts_cam, mesh_res.faces)
    _save_obj(obj_legacy_path, verts_local, mesh_res.faces)

    coord_meta = {
        "frame_name": mesh_res.frame_name,
        "model_type": mesh_res.model_type,
        "cam_t": np.asarray(mesh_res.cam_t, dtype=np.float32).reshape(3).tolist(),
        "pose_local_obj": str(obj_local_path),
        "camera_world_obj": str(obj_cam_path),
        "legacy_obj": str(obj_legacy_path),
        "coord_system": {
            "pose_local_obj": "local pose mesh (no camera translation)",
            "camera_world_obj": "camera/world metric mesh (verts + cam_t)",
        },
    }
    coord_meta_path = final_out / "last_frame_coord_meta.json"
    with coord_meta_path.open("w", encoding="utf-8") as f:
        json.dump(coord_meta, f, indent=2)

    gt_img = cv2.imread(str(mesh_res.frame_path), cv2.IMREAD_COLOR)
    if gt_img is None:
        raise FileNotFoundError(f"Cannot read GT frame image: {mesh_res.frame_path}")
    h, w = gt_img.shape[:2]

    uv, valid_v = _project_verts_to_uv(mesh_res.verts, mesh_res.cam_t, width=w, height=h, focal=5000.0)
    model_mask = _rasterize_mesh_mask(uv, valid_v, mesh_res.faces, image_hw=(h, w))
    gt_mask = _estimate_person_mask_grabcut(gt_img)

    model_cnt = _largest_contour(model_mask)
    gt_cnt = _largest_contour(gt_mask)

    overlay = gt_img.copy()
    if len(gt_cnt) > 0:
        cv2.drawContours(overlay, gt_cnt, -1, color=(0, 255, 255), thickness=2)
    if len(model_cnt) > 0:
        cv2.drawContours(overlay, model_cnt, -1, color=(0, 0, 255), thickness=2)

    gt_copy_path = final_out / f"last_frame_gt_{mesh_res.frame_name}"
    overlay_path = final_out / "last_frame_gt_model_contours.png"
    gt_mask_path = final_out / "last_frame_gt_mask.png"
    model_mask_path = final_out / "last_frame_model_mask.png"
    cv2.imwrite(str(gt_copy_path), gt_img)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(gt_mask_path), gt_mask)
    cv2.imwrite(str(model_mask_path), model_mask)

    summary = {
        "frames_dir": str(frames_dir),
        "assets_dir": str(assets_dir),
        "wham_repo_dir": str(wham_dir),
        "pymaf_repo_dir": str(pymaf_dir),
        "wham_output_dir": str(wham_out),
        "pymaf_output_dir": str(pymaf_result_dir),
        "final_output_dir": str(final_out),
        "last_frame_name": mesh_res.frame_name,
        "last_frame_obj": str(obj_legacy_path),
        "last_frame_pose_local_obj": str(obj_local_path),
        "last_frame_cam_obj": str(obj_cam_path),
        "last_frame_coord_meta": str(coord_meta_path),
        "coord_system": {
            "training_binding_recommended": "camera_world_obj",
            "editing_reference_recommended": "pose_local_obj",
        },
        "last_frame_overlay_image": str(overlay_path),
        "last_frame_gt_mask": str(gt_mask_path),
        "last_frame_model_mask": str(model_mask_path),
        "model_type": mesh_res.model_type,
        "wham_checkpoints": {k: (None if v is None else str(v)) for k, v in wham_ckpts.items()},
        "pymaf_checkpoint": (None if pymaf_ckpt is None else str(pymaf_ckpt)),
        "missing_requirements": req,
    }
    with (final_out / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Canonical stage1 mesh outputs used by downstream stage2 binding pipelines.
    frame_stem = Path(mesh_res.frame_name).stem
    stage1_tpose_obj = out_dir / "smpl_mesh_tpose.obj"
    stage1_tpose_named_obj = out_dir / f"{frame_stem}_t-pose.obj"
    stage1_seq_dir = out_dir / "smpl_mesh_sequence"
    stage1_seq_dir_cn = out_dir / "sequence"
    stage1_seq_dir.mkdir(parents=True, exist_ok=True)
    stage1_seq_dir_cn.mkdir(parents=True, exist_ok=True)
    seq_obj_path = stage1_seq_dir / f"{frame_stem}.obj"
    seq_obj_cn_path = stage1_seq_dir_cn / f"{frame_stem}.obj"
    _save_obj(stage1_tpose_obj, verts_cam, mesh_res.faces)
    _save_obj(stage1_tpose_named_obj, verts_cam, mesh_res.faces)
    _save_obj(seq_obj_path, verts_cam, mesh_res.faces)
    _save_obj(seq_obj_cn_path, verts_cam, mesh_res.faces)

    if bool(args.minimal_stage1_outputs):
        moved = {}
        # Keep only canonical objs in stage1 root; move bulky/intermediate outputs to trash (recoverable).
        for tag, pp in [('wham', wham_out), ('pymaf', pymaf_out), ('runtime', runtime_out), ('final', final_out)]:
            dst = _move_path_to_trash(pp, project_root=project_root, tag=tag)
            moved[tag] = (None if dst is None else str(dst))

        compact_summary = {
            'stage1_dir': str(out_dir),
            'frame_name': mesh_res.frame_name,
            'smpl_mesh_tpose_obj': str(stage1_tpose_obj),
            'frame_tpose_obj': str(stage1_tpose_named_obj),
            'smpl_mesh_sequence_obj': str(seq_obj_path),
            'frame_sequence_obj_en_dir': str(seq_obj_cn_path),
            'pruned_to_trash': moved,
        }
        with (out_dir / 'stage1_minimal_summary.json').open('w', encoding='utf-8') as f:
            json.dump(compact_summary, f, indent=2)

    print("Done.")
    print(f"Last frame OBJ (legacy local): {obj_legacy_path}")
    print(f"Last frame OBJ (pose local): {obj_local_path}")
    print(f"Last frame OBJ (camera/world): {obj_cam_path}")
    print(f"Coordinate metadata: {coord_meta_path}")
    print(f"Contour overlay: {overlay_path}")
    print(f"Summary: {final_out / 'run_summary.json'}")


if __name__ == "__main__":
    main()




