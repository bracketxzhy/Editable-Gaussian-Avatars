from __future__ import annotations
from pathlib import Path
import json
import cv2
import torch
from torch.utils.data import Dataset


def _read_rgb(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0  # (H,W,3)
    return x.permute(2, 0, 1).contiguous()     # (3,H,W)


class FrameDataset(Dataset):
    """
    Reads frames from preprocess output directory:
      data/frames/<name>/*.jpg
      data/frames/<name>/metadata.json
    """
    def __init__(self, frames_dir: str | Path, max_frames: int = -1, stride: int = 1):
        self.frames_dir = Path(frames_dir)
        meta_path = self.frames_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {self.frames_dir}")

        meta = json.load(meta_path.open("r", encoding="utf-8"))
        frames = meta.get("frames", [])
        if stride > 1:
            frames = frames[::stride]
        if max_frames > 0:
            frames = frames[:max_frames]

        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        item = self.frames[idx]
        img_path = self.frames_dir / item["file"]
        img = _read_rgb(img_path)  # (3,H,W)
        return {
            "image": img,
            "name": item["file"],
            "src_frame_idx": item.get("src_frame_idx", idx),
        }
