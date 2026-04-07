from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = str(Path(video_path))
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break
            yield idx, frame_bgr
            idx += 1

    def release(self) -> None:
        self.cap.release()
