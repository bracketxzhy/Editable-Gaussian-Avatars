from __future__ import annotations
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_keypoints_from_frames(
    frames_dir: str | Path,
    out_json: str | Path,
    frame_paths: list[str | Path] | None = None,
    static_image_mode: bool = True,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    frames_dir = Path(frames_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pose = mp_pose.Pose(
        static_image_mode=bool(static_image_mode),
        model_complexity=int(model_complexity),
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    all_kpts = {}

    if frame_paths is None:
        img_paths = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        img_paths = sorted(img_paths)
    else:
        img_paths = [Path(p) for p in frame_paths]

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            continue

        h, w = img.shape[:2]
        kpts = []
        for lm in res.pose_landmarks.landmark:
            kpts.append([lm.x * w, lm.y * h, lm.visibility])

        all_kpts[img_path.name] = np.array(kpts, dtype=np.float32).tolist()

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_kpts, f, indent=2)

    return out_json
