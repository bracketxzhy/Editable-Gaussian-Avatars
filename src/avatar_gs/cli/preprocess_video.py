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
    jpeg_quality = int(cfg["video"].get("jpeg_quality", 95))

    # 强制：永远保持原始分辨率（无论 YAML 写了什么 resize 参数）
    # 如果你希望未来允许 resize，再加一个 allow_resize 开关即可；但你现在的需求是“永不 resize”。
    resize_w = int(cfg["video"].get("resize_w", 0))
    resize_h = int(cfg["video"].get("resize_h", 0))
    if resize_w != 0 or resize_h != 0:
        logger.warning(
            f"[FORCE_ORIG_SIZE] resize_w/resize_h in config are ({resize_w},{resize_h}) "
            f"but will be IGNORED to keep output exactly the same size as input video."
        )
    resize_w = 0
    resize_h = 0

    logger.info(f"Video: {video_path}")
    logger.info(f"Output frames dir: {out_dir}")
    logger.info(f"frame_step={step}, max_frames={max_frames}, FORCE_ORIG_SIZE=True")

    vr = VideoReader(video_path)

    # 以 VideoReader 的 width/height 作为“必须保持的原始尺寸”
    orig_w = int(vr.width)
    orig_h = int(vr.height)

    meta = {
        "video_path": str(video_path),
        "fps": vr.fps,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "frame_step": step,
        "max_frames": max_frames,
        "resize_w": 0,
        "resize_h": 0,
        "force_orig_size": True,
        "frames": [],
    }

    saved = 0
    for idx, frame_bgr in vr:
        if idx % step != 0:
            continue
        if max_frames > 0 and saved >= max_frames:
            break

        # 关键：断言每一帧都是原始尺寸，防止上游偷偷 resize
        h, w = frame_bgr.shape[:2]
        if w != orig_w or h != orig_h:
            raise RuntimeError(
                f"Frame size mismatch at src_frame_idx={idx}: got (w,h)=({w},{h}) "
                f"but expected ({orig_w},{orig_h}). "
                f"Your VideoReader or upstream pipeline is altering resolution."
            )

        # 保存为 jpg（不改变尺寸）
        fname = f"{saved:06d}.jpg"
        fpath = out_dir / fname
        ok = cv2.imwrite(str(fpath), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to write frame: {fpath}")

        meta["frames"].append(
            {
                "index": saved,
                "file": fname,
                "src_frame_idx": idx,
                "width": w,
                "height": h,
            }
        )
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
