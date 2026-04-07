from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from ultralytics import YOLO


def _list_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(frames_dir.glob(ext))
    return sorted(out)


def _parse_indices(raw: str, n: int) -> list[int]:
    s = str(raw).strip().lower()
    if s in ("", "-1", "all", "*"):
        return list(range(n))

    chunks = s.replace(",", " ").split()
    out: list[int] = []
    for c in chunks:
        i = int(c)
        if i < 0:
            i = n + i
        if i < 0 or i >= n:
            continue
        if i not in out:
            out.append(i)

    if len(out) <= 0:
        raise ValueError(f"No valid frame indices parsed from: {raw}")
    return out


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _resolve_model_name(pretrained_model: str) -> str:
    raw = str(pretrained_model).strip()
    if raw == "":
        raise ValueError("pretrained_model is empty.")
    p = Path(raw)
    if p.is_dir():
        return p.name
    if p.suffixes[-2:] == [".pth", ".tar"]:
        return p.name[: -len(".pth.tar")]
    if p.suffix in (".pt", ".pth", ".ckpt"):
        return p.stem
    return p.name


def _bootstrap_repo_imports(project_root: Path, repo_dir: Path) -> None:
    for pp in (repo_dir, project_root / "src"):
        pp_str = str(pp.resolve())
        if pp_str not in sys.path:
            sys.path.insert(0, pp_str)


def main() -> None:
    parser = argparse.ArgumentParser("Run SMPLest-X inference on selected frame indices")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--smplestx_repo_dir", type=str, default="third_party/SMPLest-X")
    parser.add_argument("--pretrained_model", type=str, required=True, help="SMPLest-X model dir or checkpoint path")
    parser.add_argument("--frame_indices", type=str, default="-1")
    parser.add_argument("--single_person", action="store_true")
    parser.add_argument("--bbox_thr", type=int, default=50)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    frames_dir = Path(args.frames_dir).resolve()
    output_folder = Path(args.output_folder).resolve()
    smplestx_repo_dir = Path(args.smplestx_repo_dir).resolve()
    model_name = _resolve_model_name(args.pretrained_model)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not smplestx_repo_dir.exists():
        raise FileNotFoundError(f"SMPLest-X repo not found: {smplestx_repo_dir}")

    model_dir = smplestx_repo_dir / "pretrained_models" / model_name
    config_path = model_dir / "config_base.py"
    ckpt_path = model_dir / f"{model_name}.pth.tar"
    if not config_path.exists():
        raise FileNotFoundError(f"SMPLest-X config missing: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SMPLest-X checkpoint missing: {ckpt_path}")

    frames = _list_frames(frames_dir)
    if len(frames) <= 0:
        raise RuntimeError(f"No frames found under: {frames_dir}")

    selected_idx = _parse_indices(args.frame_indices, len(frames))
    selected_files = [frames[i] for i in selected_idx]

    output_folder.mkdir(parents=True, exist_ok=True)
    selected_input_dir = output_folder / "selected_frames_input"
    selected_result_dir = output_folder / "selected_frames"
    if selected_input_dir.exists():
        shutil.rmtree(selected_input_dir, ignore_errors=True)
    if selected_result_dir.exists():
        shutil.rmtree(selected_result_dir, ignore_errors=True)
    selected_input_dir.mkdir(parents=True, exist_ok=True)
    selected_result_dir.mkdir(parents=True, exist_ok=True)

    selected_input_files: list[str] = []
    for local_idx, p in enumerate(selected_files, start=1):
        dst_name = f"{local_idx:06d}{p.suffix.lower()}"
        selected_input_files.append(dst_name)
        _safe_link_or_copy(p, selected_input_dir / dst_name)

    _bootstrap_repo_imports(project_root=project_root, repo_dir=smplestx_repo_dir)
    from human_models.human_models import SMPLX
    from main.base import Tester
    from main.config import Config
    from utils.data_utils import generate_patch_image, load_img, process_bbox
    from utils.inference_utils import non_max_suppression
    try:
        from utils.visualization_utils import render_mesh
    except Exception:
        render_mesh = None

    cudnn.benchmark = True
    cfg = Config.load_config(config_path)
    cfg.update_config(
        {
            "model": {
                "pretrained_model_path": str(ckpt_path),
                "human_model_path": str((smplestx_repo_dir / "human_models" / "human_model_files").resolve()),
            },
            "inference": {
                "detection": {
                    "iou_thr": float(args.iou_thr),
                }
            },
            "log": {
                "exp_name": f"stage1_smplestx_{model_name}",
                "output_dir": str((selected_result_dir / "_logs").resolve()),
                "log_dir": str((selected_result_dir / "_logs" / "log").resolve()),
                "model_dir": None,
                "result_dir": None,
            },
        }
    )
    cfg.prepare_log()

    smpl_x = SMPLX(cfg.model.human_model_path)
    demoer = Tester(cfg)
    demoer.logger.info("Using 1 GPU.")
    demoer.logger.info(f"Inference [selected_frames] with [{ckpt_path}].")
    demoer._make_model()

    bbox_model = str(getattr(cfg.inference.detection, "model_path", "./pretrained_models/yolov8x.pt")).strip()
    bbox_model_path = Path(bbox_model)
    if not bbox_model_path.is_absolute():
        bbox_model_path = (smplestx_repo_dir / bbox_model_path).resolve()
    detector = YOLO(str(bbox_model_path))
    transform = transforms.ToTensor()

    save_path_img = selected_result_dir / "img"
    save_path_meta = selected_result_dir / "meta"
    save_path_smplx = selected_result_dir / "smplx"
    save_path_img.mkdir(parents=True, exist_ok=True)
    save_path_meta.mkdir(parents=True, exist_ok=True)
    save_path_smplx.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_name in enumerate(selected_input_files, start=1):
        img_path = selected_input_dir / frame_name
        original_img = load_img(str(img_path))
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        yolo_bbox = detector.predict(
            original_img,
            device="cuda",
            classes=0,
            conf=cfg.inference.detection.conf,
            save=cfg.inference.detection.save,
            verbose=cfg.inference.detection.verbose,
        )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            cv2.imwrite(str(save_path_img / frame_name), vis_img[:, :, ::-1])
            continue
        if bool(args.single_person):
            num_bbox = 1
        else:
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        for bbox_id in range(num_bbox):
            bbox_xywh = np.zeros((4,), dtype=np.float32)
            bbox_xywh[0] = yolo_bbox[bbox_id][0]
            bbox_xywh[1] = yolo_bbox[bbox_id][1]
            bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            if bbox_xywh[2] < float(args.bbox_thr) or bbox_xywh[3] < float(args.bbox_thr * 3):
                continue

            bbox = process_bbox(
                bbox=bbox_xywh,
                img_width=original_img_width,
                img_height=original_img_height,
                input_img_shape=cfg.model.input_img_shape,
                ratio=getattr(cfg.data, "bbox_ratio", 1.25),
            )
            img, _, _ = generate_patch_image(
                cvimg=original_img,
                bbox=bbox,
                scale=1.0,
                rot=0.0,
                do_flip=False,
                out_shape=cfg.model.input_img_shape,
            )
            img = transform(img.astype(np.float32)) / 255.0
            img = img.cuda()[None, :, :, :]
            inputs = {"img": img}

            with torch.no_grad():
                out = demoer.model(inputs, {}, {}, "test")

            mesh = out["smplx_mesh_cam"].detach().cpu().numpy()[0]
            focal = [
                cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3],
            ]
            princpt = [
                cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1],
            ]

            if render_mesh is not None:
                vis_img = render_mesh(
                    vis_img,
                    mesh,
                    smpl_x.face,
                    {"focal": focal, "princpt": princpt},
                    mesh_as_vertices=False,
                )
            vis_img = cv2.rectangle(
                vis_img,
                (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])),
                (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])),
                (0, 255, 0),
                1,
            )

            smplx_pred = {
                "global_orient": out["smplx_root_pose"].reshape(-1, 3).detach().cpu().numpy(),
                "body_pose": out["smplx_body_pose"].reshape(-1, 3).detach().cpu().numpy(),
                "left_hand_pose": out["smplx_lhand_pose"].reshape(-1, 3).detach().cpu().numpy(),
                "right_hand_pose": out["smplx_rhand_pose"].reshape(-1, 3).detach().cpu().numpy(),
                "jaw_pose": out["smplx_jaw_pose"].reshape(-1, 3).detach().cpu().numpy(),
                "leye_pose": np.zeros((1, 3), dtype=np.float32),
                "reye_pose": np.zeros((1, 3), dtype=np.float32),
                "betas": out["smplx_shape"].reshape(-1, 10).detach().cpu().numpy(),
                "expression": out["smplx_expr"].reshape(-1, 10).detach().cpu().numpy(),
                "transl": out["cam_trans"].reshape(-1, 3).detach().cpu().numpy(),
            }
            np.savez(save_path_smplx / f"{frame_idx:05d}_{bbox_id}.npz", **smplx_pred)

            meta = {
                "focal": [float(x) for x in focal],
                "princpt": [float(x) for x in princpt],
                "bbox": [float(x) for x in np.asarray(bbox).reshape(-1)],
                "bbox_detector": [float(x) for x in bbox_xywh.reshape(-1)],
                "detector": "yolo",
                "bbox_id": int(bbox_id),
                "img_path": str(img_path),
            }
            with (save_path_meta / f"{frame_idx:05d}_{bbox_id}.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        cv2.imwrite(str(save_path_img / frame_name), vis_img[:, :, ::-1])

    meta = {
        "backend": "smplestx",
        "selected_files": [p.name for p in selected_files],
        "selected_indices_global": [int(i) for i in selected_idx],
        "selected_input_files": selected_input_files,
        "frames_dir": str(frames_dir),
        "model_name": model_name,
        "single_person": bool(args.single_person),
    }
    with (selected_result_dir / "selected_frames_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"SMPLest-X inference done: {selected_result_dir}")


if __name__ == "__main__":
    main()
