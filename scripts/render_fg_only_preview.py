from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def axis_angle_to_rotmat_np(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w) + 1e-8)
    k = w / theta
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float32,
    )
    I = np.eye(3, dtype=np.float32)
    s = np.sin(theta).astype(np.float32)
    c = np.cos(theta).astype(np.float32)
    return I + s * K + (1.0 - c) * (K @ K)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render foreground-only gaussian preview (no gsplat).")
    parser.add_argument("--ckpt", type=str, default="outputs/stage2/gaussians3d.pt")
    parser.add_argument("--out", type=str, default="outputs/renders/fg_only/000000.png")
    parser.add_argument("--radius_scale", type=float, default=1.1)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    g = ckpt["gaussians"]

    xyz = g["xyz"].detach().float().cpu().numpy()  # (N,3)
    scales = torch.exp(g["log_scales"].detach()).mean(dim=1).float().cpu().numpy()  # (N,)
    opacity = torch.sigmoid(g["opacity_logit"].detach()).squeeze(-1).float().cpu().numpy()  # (N,)
    rgb = torch.sigmoid(g["rgb_logit"].detach()).float().cpu().numpy()  # (N,3), RGB

    K = torch.tensor(ckpt["K"], dtype=torch.float32).view(3, 3).cpu().numpy()
    H = int(ckpt.get("H", 1000))
    W = int(ckpt.get("W", 1000))

    cam_w = ckpt.get("cam_w", None)
    cam_t = ckpt.get("cam_t", None)
    if cam_w is None or cam_t is None:
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    else:
        w = torch.tensor(cam_w, dtype=torch.float32).view(-1, 3)[0].cpu().numpy()
        t = torch.tensor(cam_t, dtype=torch.float32).view(-1, 3)[0].cpu().numpy()
        R = axis_angle_to_rotmat_np(w)

    # Camera projection
    Xc = (xyz @ R.T) + t.reshape(1, 3)
    z = Xc[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        raise RuntimeError("No valid points in front of camera.")

    Xc = Xc[valid]
    z = z[valid]
    scales = scales[valid]
    opacity = opacity[valid]
    rgb = rgb[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (Xc[:, 0] / z) + cx
    v = fy * (Xc[:, 1] / z) + cy

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    scales = scales[in_img]
    opacity = opacity[in_img]
    rgb = rgb[in_img]

    # Draw far to near
    order = np.argsort(z)[::-1]
    u = u[order]
    v = v[order]
    z = z[order]
    scales = scales[order]
    opacity = opacity[order]
    rgb = rgb[order]

    canvas = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(u.shape[0]):
        x = int(round(float(u[i])))
        y = int(round(float(v[i])))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        # Approximate projected gaussian size in pixels
        r = int(np.clip(args.radius_scale * fx * float(scales[i]) / float(z[i]), 1.0, 8.0))
        a = float(np.clip(opacity[i], 0.02, 0.95))
        col_rgb = rgb[i]
        col_bgr = np.array([col_rgb[2], col_rgb[1], col_rgb[0]], dtype=np.float32)

        x0 = max(0, x - r)
        x1 = min(W - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(H - 1, y + r)
        if x1 <= x0 or y1 <= y0:
            continue

        patch = canvas[y0 : y1 + 1, x0 : x1 + 1]
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        w = np.exp(-0.5 * d2 / (max(1, r) ** 2)).astype(np.float32) * a
        w = w[..., None]
        patch[:] = patch * (1.0 - w) + col_bgr[None, None, :] * w

    out = (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), out)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()
