from __future__ import annotations
from pathlib import Path

import cv2

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base, save_yaml, resolve_path
from avatar_gs.utils.io import ensure_dir, write_json
from avatar_gs.utils.log import setup_logger
from avatar_gs.utils.seed import seed_everything
from avatar_gs.data.video_reader import VideoReader


def main():
    parser = parse_args_base("Preprocess monocular video: extract frames + write metadata")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    project_root = Path(cfg.get("project_root", ".")).resolve()
    video_path = Path(resolve_path(cfg["video"]["path"]))
    out_dir = project_root / cfg["video"]["out_frames_dir"]
    out_dir = ensure_dir(out_dir)

    log_dir = ensure_dir(project_root / cfg.get("log_dir", "outputs/logs"))
    logger = setup_logger("preprocess_video", str(log_dir / "preprocess_video.log"))

    step = int(cfg["video"].get("frame_step", 1))
    max_frames = int(cfg["video"].get("max_frames", -1))
    resize_w = int(cfg["video"].get("resize_w", 0))
    resize_h = int(cfg["video"].get("resize_h", 0))
    jpeg_quality = int(cfg["video"].get("jpeg_quality", 95))

    logger.info(f"Video: {video_path}")
    logger.info(f"Output frames dir: {out_dir}")
    logger.info(f"frame_step={step}, max_frames={max_frames}, resize=({resize_w},{resize_h})")

    vr = VideoReader(video_path)
    meta = {
        "video_path": str(video_path),
        "fps": vr.fps,
        "orig_width": vr.width,
        "orig_height": vr.height,
        "frame_step": step,
        "max_frames": max_frames,
        "resize_w": resize_w,
        "resize_h": resize_h,
        "frames": []
    }

    saved = 0
    for idx, frame_bgr in vr:
        if idx % step != 0:
            continue
        if max_frames > 0 and saved >= max_frames:
            break

        if resize_w > 0 and resize_h > 0:
            frame_bgr = cv2.resize(frame_bgr, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

        # 保存为 jpg
        fname = f"{saved:06d}.jpg"
        fpath = out_dir / fname
        ok = cv2.imwrite(str(fpath), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to write frame: {fpath}")

        meta["frames"].append({"index": saved, "file": fname, "src_frame_idx": idx})
        saved += 1

        if saved % 50 == 0:
            logger.info(f"Saved {saved} frames...")

    vr.release()

    meta_path = out_dir / "metadata.json"
    write_json(meta, meta_path)
    logger.info(f"Done. Saved frames: {saved}. Metadata: {meta_path}")

    # 保存 resolve 后的 config 快照（方便复现）
    cfg_out = project_root / cfg.get("log_dir", "outputs/logs") / "preprocess_video_resolved.yaml"
    save_yaml(cfg, cfg_out)


if __name__ == "__main__":
    main()
