from __future__ import annotations
from pathlib import Path
import json
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

POSE_KPTS = 33
HAND_KPTS = 21
LEFT_HAND_OFFSET = POSE_KPTS
RIGHT_HAND_OFFSET = POSE_KPTS + HAND_KPTS
FULL_KPTS_WITH_HANDS = POSE_KPTS + HAND_KPTS + HAND_KPTS


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (0.5 * float(bbox[0] + bbox[2]), 0.5 * float(bbox[1] + bbox[3]))


def _bbox_side(bbox: tuple[float, float, float, float]) -> float:
    return max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))


def _make_square_bbox_from_center(
    center_x: float,
    center_y: float,
    side: float,
    w: int,
    h: int,
    min_size_px: float,
) -> tuple[int, int, int, int]:
    side = max(float(side), float(max(min_size_px, 16.0)))
    left = float(center_x) - 0.5 * side
    top = float(center_y) - 0.5 * side
    max_left = max(float(w) - side, 0.0)
    max_top = max(float(h) - side, 0.0)
    left = float(np.clip(left, 0.0, max_left))
    top = float(np.clip(top, 0.0, max_top))
    right = min(left + side, float(w))
    bottom = min(top + side, float(h))

    x0i = int(np.floor(left))
    y0i = int(np.floor(top))
    x1i = int(np.ceil(right))
    y1i = int(np.ceil(bottom))
    x0i = max(0, min(x0i, w - 1))
    y0i = max(0, min(y0i, h - 1))
    x1i = max(x0i + 1, min(x1i, w))
    y1i = max(y0i + 1, min(y1i, h))
    return x0i, y0i, x1i, y1i


def _alpha(cutoff: float, fps: float) -> float:
    if fps <= 1e-6:
        return 1.0
    r = 2.0 * np.pi * float(cutoff) / float(fps)
    return float(r / (1.0 + r))


def _one_euro_smooth(
    arr: np.ndarray,          # (T,K,D)
    valid: np.ndarray,        # (T,K) bool
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    T, K, D = out.shape
    x_prev = np.zeros((K, D), dtype=np.float32)
    dx_prev = np.zeros((K, D), dtype=np.float32)
    has_prev = np.zeros((K,), dtype=np.bool_)
    for t in range(T):
        x = out[t]
        m = valid[t]
        for k in range(K):
            if not bool(m[k]):
                continue
            if not bool(has_prev[k]):
                x_prev[k] = x[k]
                dx_prev[k].fill(0.0)
                has_prev[k] = True
                continue
            dx = (x[k] - x_prev[k]) * float(fps)
            a_d = _alpha(float(d_cutoff), float(fps))
            dx_hat = dx_prev[k] + a_d * (dx - dx_prev[k])
            cutoff = float(min_cutoff) + float(beta) * float(np.linalg.norm(dx_hat))
            a = _alpha(cutoff, float(fps))
            x_hat = x_prev[k] + a * (x[k] - x_prev[k])
            out[t, k] = x_hat
            x_prev[k] = x_hat
            dx_prev[k] = dx_hat
    return out


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    ks = max(3, int(kernel_size))
    if ks % 2 == 0:
        ks += 1
    x = np.arange(ks, dtype=np.float32) - float(ks // 2)
    g = np.exp(-(x * x) / max(2.0 * float(sigma) * float(sigma), 1e-6))
    g = g / max(float(g.sum()), 1e-8)
    return g.astype(np.float32)


def _gaussian_smooth(
    arr: np.ndarray,      # (T,K,D)
    valid: np.ndarray,    # (T,K)
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    T, K, D = out.shape
    ker = _gaussian_kernel_1d(kernel_size=kernel_size, sigma=sigma)
    pad = int(len(ker) // 2)
    for k in range(K):
        vk = valid[:, k].astype(np.float32)
        if float(vk.sum()) < 2.0:
            continue
        for d in range(D):
            x = out[:, k, d]
            x0 = np.pad(x, (pad, pad), mode="edge")
            v0 = np.pad(vk, (pad, pad), mode="edge")
            num = np.convolve(x0 * v0, ker, mode="valid")
            den = np.convolve(v0, ker, mode="valid")
            den = np.clip(den, 1e-6, None)
            out[:, k, d] = num / den
    return out


def _smooth_keypoints_dict(
    all_kpts: dict[str, list[list[float]]],
    enable: bool,
    method: str,
    fps: float,
    min_cutoff: float,
    beta: float,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
) -> dict[str, list[list[float]]]:
    if not bool(enable) or len(all_kpts) <= 1:
        return all_kpts

    frame_names = list(all_kpts.keys())
    arrs = [np.asarray(all_kpts[n], dtype=np.float32) for n in frame_names]
    if len(arrs) == 0:
        return all_kpts
    K = int(arrs[0].shape[0])
    D = int(arrs[0].shape[1])
    if K <= 0 or D < 2:
        return all_kpts
    if any((a.shape[0] != K or a.shape[1] != D) for a in arrs):
        return all_kpts

    stack = np.stack(arrs, axis=0).astype(np.float32)  # (T,K,D)
    valid = stack[:, :, 2] > 1e-4
    xy = stack[:, :, :2]
    m = str(method).strip().lower()
    if m == "gaussian":
        xy_s = _gaussian_smooth(
            arr=xy,
            valid=valid,
            kernel_size=int(gaussian_kernel_size),
            sigma=float(gaussian_sigma),
        )
    else:
        xy_s = _one_euro_smooth(
            arr=xy,
            valid=valid,
            fps=float(max(1.0, fps)),
            min_cutoff=float(max(1e-3, min_cutoff)),
            beta=float(max(0.0, beta)),
            d_cutoff=1.0,
        )
    stack[:, :, :2] = xy_s
    return {name: stack[i].tolist() for i, name in enumerate(frame_names)}


def _bbox_union(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None):
    if a is None:
        return b
    if b is None:
        return a
    return (
        float(min(a[0], b[0])),
        float(min(a[1], b[1])),
        float(max(a[2], b[2])),
        float(max(a[3], b[3])),
    )


def _bbox_from_landmarks(res, w: int, h: int, min_visibility: float = 0.1) -> tuple[float, float, float, float] | None:
    if res is None or getattr(res, "pose_landmarks", None) is None:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for lm in res.pose_landmarks.landmark:
        if float(getattr(lm, "visibility", 0.0)) < float(min_visibility):
            continue
        x = float(lm.x) * float(w)
        y = float(lm.y) * float(h)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 4:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_from_segmentation(seg: np.ndarray | None, threshold: float) -> tuple[float, float, float, float] | None:
    if seg is None:
        return None
    m = np.asarray(seg, dtype=np.float32) >= float(threshold)
    if int(m.sum()) < 32:
        return None
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _make_square_bbox(
    bbox: tuple[float, float, float, float],
    w: int,
    h: int,
    scale: float,
    min_size_px: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(float(x1 - x0), 1.0)
    bh = max(float(y1 - y0), 1.0)
    side = max(bw, bh) * float(max(scale, 1.0))
    side = max(side, float(max(min_size_px, 16.0)))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    left = cx - 0.5 * side
    top = cy - 0.5 * side
    max_left = max(float(w) - side, 0.0)
    max_top = max(float(h) - side, 0.0)
    left = float(np.clip(left, 0.0, max_left))
    top = float(np.clip(top, 0.0, max_top))
    right = min(left + side, float(w))
    bottom = min(top + side, float(h))

    x0i = int(np.floor(left))
    y0i = int(np.floor(top))
    x1i = int(np.ceil(right))
    y1i = int(np.ceil(bottom))
    x0i = max(0, min(x0i, w - 1))
    y0i = max(0, min(y0i, h - 1))
    x1i = max(x0i + 1, min(x1i, w))
    y1i = max(y0i + 1, min(y1i, h))
    return x0i, y0i, x1i, y1i


def _landmarks_to_kpts(res, w: int, h: int, offset_x: float = 0.0, offset_y: float = 0.0):
    world_lms = getattr(res, "pose_world_landmarks", None)
    world_seq = world_lms.landmark if world_lms is not None else None
    kpts = []
    for li, lm in enumerate(res.pose_landmarks.landmark):
        x = float(lm.x) * float(w) + float(offset_x)
        y = float(lm.y) * float(h) + float(offset_y)
        vis = float(getattr(lm, "visibility", 0.0))
        z = float(getattr(lm, "z", 0.0))
        if world_seq is not None and li < len(world_seq):
            wlm = world_seq[li]
            kpts.append([x, y, vis, z, float(wlm.x), float(wlm.y), float(wlm.z)])
        else:
            kpts.append([x, y, vis, z, 0.0, 0.0, 0.0])
    return np.asarray(kpts, dtype=np.float32)


def _empty_hand_kpts() -> np.ndarray:
    return np.zeros((HAND_KPTS, 7), dtype=np.float32)


def _hand_landmarks_to_kpts(hand_lms, w: int, h: int, score: float = 1.0) -> np.ndarray:
    """
    Convert MediaPipe Hands landmarks to (21,7):
      [x, y, conf, z, wx, wy, wz]
    Hand world coordinates are not used in current stage1, so set wx/wy/wz=0.
    """
    out = np.zeros((HAND_KPTS, 7), dtype=np.float32)
    conf = float(np.clip(score, 0.0, 1.0))
    seq = getattr(hand_lms, "landmark", [])
    for i in range(min(HAND_KPTS, len(seq))):
        lm = seq[i]
        out[i, 0] = float(lm.x) * float(w)
        out[i, 1] = float(lm.y) * float(h)
        out[i, 2] = conf
        # MediaPipe hand z is weakly defined and not metric-consistent with pose z.
        out[i, 3] = 0.0
    return out


def _assign_hands_to_lr(
    hand_candidates: list[dict],
    pose_kpts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = _empty_hand_kpts()
    right = _empty_hand_kpts()
    if len(hand_candidates) == 0:
        return left, right

    used = set()
    lw_valid = pose_kpts.shape[0] > 15 and float(pose_kpts[15, 2]) > 0.05
    rw_valid = pose_kpts.shape[0] > 16 and float(pose_kpts[16, 2]) > 0.05
    lw = pose_kpts[15, :2] if lw_valid else None
    rw = pose_kpts[16, :2] if rw_valid else None

    if lw is not None and rw is not None:
        # Prefer wrist-distance assignment when both wrists are visible.
        order = sorted(
            range(len(hand_candidates)),
            key=lambda i: float(
                min(
                    np.linalg.norm(hand_candidates[i]["wrist"] - lw),
                    np.linalg.norm(hand_candidates[i]["wrist"] - rw),
                )
            ),
        )
        left_taken = False
        right_taken = False
        for i in order:
            d_l = float(np.linalg.norm(hand_candidates[i]["wrist"] - lw))
            d_r = float(np.linalg.norm(hand_candidates[i]["wrist"] - rw))
            prefer_left = d_l <= d_r
            if prefer_left and not left_taken:
                left = hand_candidates[i]["kpts"]
                left_taken = True
                used.add(i)
            elif (not prefer_left) and not right_taken:
                right = hand_candidates[i]["kpts"]
                right_taken = True
                used.add(i)
            elif not left_taken:
                left = hand_candidates[i]["kpts"]
                left_taken = True
                used.add(i)
            elif not right_taken:
                right = hand_candidates[i]["kpts"]
                right_taken = True
                used.add(i)
            if left_taken and right_taken:
                break

    # Fill remaining slots by handedness label.
    for i, cand in enumerate(hand_candidates):
        if i in used:
            continue
        label = str(cand.get("label", "")).lower()
        if label == "left" and float(left[:, 2].max()) <= 0.0:
            left = cand["kpts"]
            used.add(i)
        elif label == "right" and float(right[:, 2].max()) <= 0.0:
            right = cand["kpts"]
            used.add(i)

    # Last fallback: fill empty side with any remaining hand.
    for i, cand in enumerate(hand_candidates):
        if i in used:
            continue
        if float(left[:, 2].max()) <= 0.0:
            left = cand["kpts"]
            used.add(i)
        elif float(right[:, 2].max()) <= 0.0:
            right = cand["kpts"]
            used.add(i)
        if float(left[:, 2].max()) > 0.0 and float(right[:, 2].max()) > 0.0:
            break

    return left, right


def extract_keypoints_from_frames(
    frames_dir: str | Path,
    out_json: str | Path,
    frame_paths: list[str | Path] | None = None,
    static_image_mode: bool = True,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    enable_segmentation: bool = True,
    segmentation_threshold: float = 0.5,
    two_stage_crop: bool = False,
    two_stage_crop_scale: float = 2.0,
    two_stage_min_size_px: float = 160.0,
    two_stage_min_visibility: float = 0.1,
    two_stage_use_prev_bbox: bool = True,
    two_stage_center_on_mask: bool = True,
    two_stage_crop_stable_momentum: float = 0.65,
    save_segmentation_npz: str | Path | None = None,
    save_crop_metadata_npz: str | Path | None = None,
    enable_hand_landmarks: bool = True,
    hand_max_num_hands: int = 2,
    hand_model_complexity: int = 1,
    hand_min_detection_confidence: float = 0.35,
    hand_min_tracking_confidence: float = 0.45,
    keypoint_smooth_enable: bool = False,
    keypoint_smooth_method: str = "one_euro",
    keypoint_smooth_fps: float = 30.0,
    keypoint_smooth_min_cutoff: float = 1.0,
    keypoint_smooth_beta: float = 0.3,
    keypoint_smooth_gaussian_kernel_size: int = 7,
    keypoint_smooth_gaussian_sigma: float = 1.5,
):
    frames_dir = Path(frames_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pose = mp_pose.Pose(
        static_image_mode=bool(static_image_mode),
        model_complexity=int(model_complexity),
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
        enable_segmentation=bool(enable_segmentation),
    )
    hands = None
    if bool(enable_hand_landmarks):
        hands = mp_hands.Hands(
            static_image_mode=bool(static_image_mode),
            max_num_hands=max(1, int(hand_max_num_hands)),
            model_complexity=max(0, int(hand_model_complexity)),
            min_detection_confidence=float(hand_min_detection_confidence),
            min_tracking_confidence=float(hand_min_tracking_confidence),
        )
    all_kpts = {}
    seg_frame_names: list[str] = []
    seg_masks: list[np.ndarray] = []
    seg_h: int | None = None
    seg_w: int | None = None
    prev_bbox: tuple[int, int, int, int] | None = None
    prev_crop_center_side: tuple[float, float, float] | None = None
    crop_meta: dict[str, dict[str, np.ndarray | float | str]] = {}

    def _stable_crop_box(
        bbox_union: tuple[float, float, float, float] | None,
        bbox_mask: tuple[float, float, float, float] | None,
        w: int,
        h: int,
    ) -> tuple[tuple[int, int, int, int] | None, str]:
        nonlocal prev_crop_center_side
        if bbox_union is None:
            return None, "none"
        source_bbox = bbox_mask if (bool(two_stage_center_on_mask) and bbox_mask is not None) else bbox_union
        source = "mask" if (source_bbox is bbox_mask and bbox_mask is not None) else "union"
        cx, cy = _bbox_center(source_bbox)
        side = _bbox_side(bbox_union) * float(max(two_stage_crop_scale, 1.0))
        side = max(side, float(max(two_stage_min_size_px, 16.0)))

        mom = float(np.clip(two_stage_crop_stable_momentum, 0.0, 0.98))
        if prev_crop_center_side is not None and mom > 0.0:
            pcx, pcy, pside = prev_crop_center_side
            cx = mom * float(pcx) + (1.0 - mom) * float(cx)
            cy = mom * float(pcy) + (1.0 - mom) * float(cy)
            side = mom * float(pside) + (1.0 - mom) * float(side)
            source = "stable"
        box = _make_square_bbox_from_center(
            center_x=float(cx),
            center_y=float(cy),
            side=float(side),
            w=int(w),
            h=int(h),
            min_size_px=float(two_stage_min_size_px),
        )
        prev_crop_center_side = (float(cx), float(cy), float(side))
        return box, source

    if frame_paths is None:
        img_paths = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
        img_paths = sorted(img_paths)
    else:
        img_paths = [Path(p) for p in frame_paths]

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_box_used = (0, 0, int(w), int(h))
        crop_source = "full"

        res = pose.process(rgb)
        if (not getattr(res, "pose_landmarks", None)) and bool(two_stage_crop) and prev_bbox is not None:
            x0, y0, x1, y1 = prev_bbox
            crop = img[y0:y1, x0:x1]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res_crop = pose.process(crop_rgb)
                if getattr(res_crop, "pose_landmarks", None):
                    crop_box_used = (int(x0), int(y0), int(x1), int(y1))
                    crop_source = "prev"
                    res = res_crop
                    kpts = _landmarks_to_kpts(res_crop, w=(x1 - x0), h=(y1 - y0), offset_x=float(x0), offset_y=float(y0))
                    all_kpts[img_path.name] = kpts.tolist()
                    if bool(enable_segmentation) and getattr(res_crop, "segmentation_mask", None) is not None:
                        seg_crop = np.asarray(res_crop.segmentation_mask, dtype=np.float32)
                        seg = np.zeros((h, w), dtype=np.float32)
                        seg[y0:y1, x0:x1] = seg_crop
                        seg_bin = (seg >= float(segmentation_threshold)).astype(np.uint8)
                    else:
                        seg_bin = None
                    lm_bbox = _bbox_from_landmarks(res_crop, w=(x1 - x0), h=(y1 - y0), min_visibility=float(two_stage_min_visibility))
                    if lm_bbox is not None:
                        lm_bbox = (
                            lm_bbox[0] + float(x0),
                            lm_bbox[1] + float(y0),
                            lm_bbox[2] + float(x0),
                            lm_bbox[3] + float(y0),
                        )
                        prev_bbox, _ = _stable_crop_box(
                            bbox_union=lm_bbox,
                            bbox_mask=None,
                            w=int(w),
                            h=int(h),
                        )
                    else:
                        prev_bbox = None
                    if seg_bin is not None:
                        if seg_h is None or seg_w is None:
                            seg_h, seg_w = int(seg_bin.shape[0]), int(seg_bin.shape[1])
                        elif int(seg_bin.shape[0]) != int(seg_h) or int(seg_bin.shape[1]) != int(seg_w):
                            seg_bin = cv2.resize(seg_bin, (int(seg_w), int(seg_h)), interpolation=cv2.INTER_NEAREST)
                        seg_frame_names.append(img_path.name)
                        seg_masks.append(seg_bin)
                    # Hand landmarks are extracted on full-frame rgb for stability.
                    if hands is not None and kpts is not None:
                        hres = hands.process(rgb)
                        cands = []
                        multi_lms = getattr(hres, "multi_hand_landmarks", None)
                        multi_handedness = getattr(hres, "multi_handedness", None)
                        if multi_lms is not None:
                            for hi, hlm in enumerate(multi_lms):
                                score = 1.0
                                label = ""
                                if multi_handedness is not None and hi < len(multi_handedness):
                                    cls = multi_handedness[hi].classification
                                    if len(cls) > 0:
                                        score = float(getattr(cls[0], "score", 1.0))
                                        label = str(getattr(cls[0], "label", ""))
                                hk = _hand_landmarks_to_kpts(hlm, w=w, h=h, score=score)
                                cands.append(
                                    {
                                        "kpts": hk,
                                        "wrist": hk[0, :2].copy(),
                                        "label": label,
                                    }
                                )
                        lh, rh = _assign_hands_to_lr(cands, kpts)
                        kpts = np.concatenate([kpts[:POSE_KPTS], lh, rh], axis=0)
                    all_kpts[img_path.name] = kpts.tolist()
                    crop_meta[img_path.name] = {
                        "crop_box_xyxy": np.asarray(crop_box_used, dtype=np.int32),
                        "image_hw": np.asarray([int(h), int(w)], dtype=np.int32),
                        "crop_scale_rel": float(max(crop_box_used[2] - crop_box_used[0], crop_box_used[3] - crop_box_used[1]) / max(float(max(h, w)), 1.0)),
                        "crop_source": crop_source,
                    }
                    continue

        if not getattr(res, "pose_landmarks", None):
            prev_bbox = None if not bool(two_stage_use_prev_bbox) else prev_bbox
            continue

        kpts = None
        seg_full = np.asarray(res.segmentation_mask, dtype=np.float32) if getattr(res, "segmentation_mask", None) is not None else None

        if bool(two_stage_crop):
            bbox_lm = _bbox_from_landmarks(
                res,
                w=w,
                h=h,
                min_visibility=float(two_stage_min_visibility),
            )
            bbox_seg = _bbox_from_segmentation(seg_full, threshold=float(segmentation_threshold))
            bbox = _bbox_union(bbox_lm, bbox_seg)
            if bbox is None and bool(two_stage_use_prev_bbox):
                bbox = prev_bbox
            if bbox is not None:
                box_cur, box_source = _stable_crop_box(
                    bbox_union=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    bbox_mask=bbox_seg,
                    w=int(w),
                    h=int(h),
                )
                if box_cur is None:
                    prev_bbox = None if not bool(two_stage_use_prev_bbox) else prev_bbox
                    box_cur = (0, 0, int(w), int(h))
                    box_source = "none"
                x0, y0, x1, y1 = box_cur
                crop_box_used = (int(x0), int(y0), int(x1), int(y1))
                crop_source = str(box_source)
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res_crop = pose.process(crop_rgb)
                    if getattr(res_crop, "pose_landmarks", None):
                        kpts = _landmarks_to_kpts(res_crop, w=(x1 - x0), h=(y1 - y0), offset_x=float(x0), offset_y=float(y0))
                        if bool(enable_segmentation) and getattr(res_crop, "segmentation_mask", None) is not None:
                            seg_crop = np.asarray(res_crop.segmentation_mask, dtype=np.float32)
                            seg_full = np.zeros((h, w), dtype=np.float32)
                            seg_full[y0:y1, x0:x1] = seg_crop
                        bbox_next = _bbox_from_landmarks(
                            res_crop,
                            w=(x1 - x0),
                            h=(y1 - y0),
                            min_visibility=float(two_stage_min_visibility),
                        )
                        if bbox_next is not None:
                            bbox_next = (
                                bbox_next[0] + float(x0),
                                bbox_next[1] + float(y0),
                                bbox_next[2] + float(x0),
                                bbox_next[3] + float(y0),
                            )
                            prev_bbox, _ = _stable_crop_box(
                                bbox_union=bbox_next,
                                bbox_mask=None,
                                w=int(w),
                                h=int(h),
                            )
                        else:
                            prev_bbox = (x0, y0, x1, y1) if bool(two_stage_use_prev_bbox) else None
                    else:
                        prev_bbox = (x0, y0, x1, y1) if bool(two_stage_use_prev_bbox) else None
                else:
                    prev_bbox = None if not bool(two_stage_use_prev_bbox) else prev_bbox
            else:
                prev_bbox = None if not bool(two_stage_use_prev_bbox) else prev_bbox

        if kpts is None:
            kpts = _landmarks_to_kpts(res, w=w, h=h)
            bbox_next = _bbox_from_landmarks(
                res,
                w=w,
                h=h,
                min_visibility=float(two_stage_min_visibility),
            )
            if bbox_next is not None:
                prev_bbox, _ = _stable_crop_box(
                    bbox_union=bbox_next,
                    bbox_mask=None,
                    w=int(w),
                    h=int(h),
                )

        if hands is not None and kpts is not None:
            hres = hands.process(rgb)
            cands = []
            multi_lms = getattr(hres, "multi_hand_landmarks", None)
            multi_handedness = getattr(hres, "multi_handedness", None)
            if multi_lms is not None:
                for hi, hlm in enumerate(multi_lms):
                    score = 1.0
                    label = ""
                    if multi_handedness is not None and hi < len(multi_handedness):
                        cls = multi_handedness[hi].classification
                        if len(cls) > 0:
                            score = float(getattr(cls[0], "score", 1.0))
                            label = str(getattr(cls[0], "label", ""))
                    hk = _hand_landmarks_to_kpts(hlm, w=w, h=h, score=score)
                    cands.append(
                        {
                            "kpts": hk,
                            "wrist": hk[0, :2].copy(),
                            "label": label,
                        }
                    )
            lh, rh = _assign_hands_to_lr(cands, kpts)
            kpts = np.concatenate([kpts[:POSE_KPTS], lh, rh], axis=0)
        all_kpts[img_path.name] = kpts.tolist()
        crop_meta[img_path.name] = {
            "crop_box_xyxy": np.asarray(crop_box_used, dtype=np.int32),
            "image_hw": np.asarray([int(h), int(w)], dtype=np.int32),
            "crop_scale_rel": float(max(crop_box_used[2] - crop_box_used[0], crop_box_used[3] - crop_box_used[1]) / max(float(max(h, w)), 1.0)),
            "crop_source": crop_source,
        }
        if bool(enable_segmentation) and seg_full is not None:
            seg_bin = (seg_full >= float(segmentation_threshold)).astype(np.uint8)
            if seg_h is None or seg_w is None:
                seg_h, seg_w = int(seg_bin.shape[0]), int(seg_bin.shape[1])
            elif int(seg_bin.shape[0]) != int(seg_h) or int(seg_bin.shape[1]) != int(seg_w):
                seg_bin = cv2.resize(seg_bin, (int(seg_w), int(seg_h)), interpolation=cv2.INTER_NEAREST)
            seg_frame_names.append(img_path.name)
            seg_masks.append(seg_bin)

    all_kpts = _smooth_keypoints_dict(
        all_kpts=all_kpts,
        enable=bool(keypoint_smooth_enable),
        method=str(keypoint_smooth_method),
        fps=float(keypoint_smooth_fps),
        min_cutoff=float(keypoint_smooth_min_cutoff),
        beta=float(keypoint_smooth_beta),
        gaussian_kernel_size=int(keypoint_smooth_gaussian_kernel_size),
        gaussian_sigma=float(keypoint_smooth_gaussian_sigma),
    )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_kpts, f, indent=2)

    if save_segmentation_npz is not None and len(seg_masks) > 0:
        seg_npz = Path(save_segmentation_npz)
        seg_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            seg_npz,
            frame_names=np.asarray(seg_frame_names),
            masks=np.stack(seg_masks, axis=0).astype(np.uint8),
        )

    if save_crop_metadata_npz is not None and len(crop_meta) > 0:
        meta_npz = Path(save_crop_metadata_npz)
        meta_npz.parent.mkdir(parents=True, exist_ok=True)
        frame_names_meta = [str(k) for k in crop_meta.keys()]
        boxes = np.stack([crop_meta[k]["crop_box_xyxy"] for k in frame_names_meta], axis=0).astype(np.int32)
        image_hw = np.stack([crop_meta[k]["image_hw"] for k in frame_names_meta], axis=0).astype(np.int32)
        crop_scale_rel = np.asarray([float(crop_meta[k]["crop_scale_rel"]) for k in frame_names_meta], dtype=np.float32)
        crop_source = np.asarray([str(crop_meta[k]["crop_source"]) for k in frame_names_meta])
        np.savez_compressed(
            meta_npz,
            frame_names=np.asarray(frame_names_meta),
            crop_boxes_xyxy=boxes,
            image_hw=image_hw,
            crop_scale_rel=crop_scale_rel,
            crop_source=crop_source,
        )

    try:
        pose.close()
    except Exception:
        pass
    if hands is not None:
        try:
            hands.close()
        except Exception:
            pass

    return out_json
