from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import struct
import torch

from avatar_gs.gaussian.structured_gaussian import Gaussian3D
from avatar_gs.gaussian.init_from_smpl import load_obj_vertices


def _load_ckpt(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    gsd = ckpt.get("gaussians") or ckpt.get("gaussian") or ckpt.get("g")
    if gsd is None:
        raise KeyError(f"Checkpoint missing gaussians. Keys: {list(ckpt.keys())}")
    g = Gaussian3D.load_state_dict(gsd)
    return g, ckpt


def _resolve_mesh_index(
    ckpt: dict,
    frame_idx: int,
    seq_npz: Path | None,
) -> int:
    if seq_npz is None:
        return frame_idx

    if not seq_npz.exists():
        raise FileNotFoundError(f"seq_npz not found: {seq_npz}")

    frames = ckpt.get("frames", None)
    if frames is None or frame_idx >= len(frames):
        return frame_idx

    seq = np.load(str(seq_npz), allow_pickle=True)
    seq_names = [str(x) for x in seq["frame_names"].tolist()]
    name_to_idx = {n: i for i, n in enumerate(seq_names)}
    name = str(frames[frame_idx])
    return int(name_to_idx.get(name, frame_idx))


def _write_ply_points(path: Path, xyz: np.ndarray, rgb: np.ndarray, alpha: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    has_alpha = alpha is not None
    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    if has_alpha:
        a_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8).reshape(-1)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if has_alpha:
            f.write("property uchar alpha\n")
        f.write("end_header\n")
        if has_alpha:
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb_u8[i]
                a = int(a_u8[i])
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {a}\n")
        else:
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb_u8[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def _write_ply_3dgs(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    log_scales: np.ndarray,
    quat: np.ndarray,
    sh_degree: int = 3,
    binary: bool = True,
) -> None:
    """
    Write a 3DGS-style PLY compatible with common viewers.
    Properties:
      x y z
      nx ny nz
      f_dc_0..2
      f_rest_0..(3*((sh_degree+1)^2-1)-1)
      opacity
      scale_0..2
      rot_0..3
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    # SH DC coefficient: color = f_dc * 0.282095
    sh0 = rgb / 0.282095
    # f_rest zeros
    n_rest = 3 * (((sh_degree + 1) ** 2) - 1)
    f_rest = np.zeros((n, n_rest), dtype=np.float32)

    mode = "wb" if binary else "w"
    with path.open(mode) as f:
        f.write("ply\n")
        f.write("format binary_little_endian 1.0\n" if binary else "format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n")
        for i in range(n_rest):
            f.write(f"property float f_rest_{i}\n")
        f.write("property float opacity\n")
        for i in range(3):
            f.write(f"property float scale_{i}\n")
        for i in range(4):
            f.write(f"property float rot_{i}\n")
        f.write("end_header\n")

        if binary:
            # Pack floats in little-endian
            fmt = "<" + "f" * (3 + 3 + 3 + n_rest + 1 + 3 + 4)
            for i in range(n):
                x, y, z = xyz[i]
                nx = ny = nz = 0.0
                fdc = sh0[i]
                fr = f_rest[i]
                op = float(opacity[i])
                sc = log_scales[i]
                q = quat[i]
                data = [
                    float(x), float(y), float(z),
                    float(nx), float(ny), float(nz),
                    float(fdc[0]), float(fdc[1]), float(fdc[2]),
                    *[float(v) for v in fr],
                    float(op),
                    float(sc[0]), float(sc[1]), float(sc[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                ]
                f.write(struct.pack(fmt, *data))
        else:
            for i in range(n):
                x, y, z = xyz[i]
                # normals are unused by most 3DGS viewers
                nx = ny = nz = 0.0
                fdc = sh0[i]
                fr = f_rest[i]
                op = float(opacity[i])
                sc = log_scales[i]
                q = quat[i]
                f.write(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{nx:.6f} {ny:.6f} {nz:.6f} "
                    f"{fdc[0]:.6f} {fdc[1]:.6f} {fdc[2]:.6f} "
                )
                f.write(" ".join(f"{v:.6f}" for v in fr))
                f.write(
                    f" {op:.6f} "
                    f"{sc[0]:.6f} {sc[1]:.6f} {sc[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
                )


def main():
    ap = argparse.ArgumentParser(description="Export gaussians to PLY for Blender preview.")
    ap.add_argument("--ckpt", required=True, type=Path, help="Stage2 gaussian checkpoint (.pt)")
    ap.add_argument("--out", required=True, type=Path, help="Output PLY path")
    ap.add_argument("--frame-idx", type=int, default=0, help="Frame index to export")
    ap.add_argument("--bind-mesh-dir", type=Path, default=None, help="SMPL mesh sequence dir (smpl_mesh_sequence)")
    ap.add_argument("--bind-seq", type=Path, default=None, help="smpl_params_sequence.npz to map frame names")
    ap.add_argument("--format", choices=("points", "3dgs"), default="points", help="Export format")
    ap.add_argument("--no-alpha", action="store_true", help="Do not write alpha channel (points only)")
    ap.add_argument("--opacity-mode", choices=("logit", "sigmoid"), default="logit", help="Opacity encoding (3dgs)")
    ap.add_argument("--scale-mode", choices=("log", "linear"), default="log", help="Scale encoding (3dgs)")
    ap.add_argument("--sh-degree", type=int, default=3, help="SH degree for 3dgs export")
    ap.add_argument("--binary", action="store_true", help="Write binary little-endian PLY (3dgs)")
    args = ap.parse_args()

    g, ckpt = _load_ckpt(args.ckpt)

    # Resolve xyz (bound or static)
    if args.bind_mesh_dir is not None and g.bind_idx is not None:
        mesh_idx = _resolve_mesh_index(ckpt, int(args.frame_idx), args.bind_seq)
        obj_path = args.bind_mesh_dir / f"{mesh_idx:06d}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"Mesh frame not found: {obj_path}")
        verts = load_obj_vertices(obj_path)
        bind_idx = g.bind_idx.detach().cpu().numpy().astype(np.int64)
        xyz = verts[bind_idx]
    else:
        xyz = g.xyz.detach().cpu().numpy()

    rgb = torch.sigmoid(g.rgb_logit).detach().cpu().numpy()

    if args.format == "points":
        alpha = None if args.no_alpha else torch.sigmoid(g.opacity_logit).detach().cpu().numpy()
        _write_ply_points(args.out, xyz, rgb, alpha)
    else:
        if args.opacity_mode == "sigmoid":
            opacity = torch.sigmoid(g.opacity_logit).detach().cpu().numpy().reshape(-1)
        else:
            opacity = g.opacity_logit.detach().cpu().numpy().reshape(-1)
        if args.scale_mode == "linear":
            log_scales = torch.exp(g.log_scales).detach().cpu().numpy()
        else:
            log_scales = g.log_scales.detach().cpu().numpy()
        quat = g.normalized_quat().detach().cpu().numpy()
        _write_ply_3dgs(
            args.out,
            xyz=xyz,
            rgb=rgb,
            opacity=opacity,
            log_scales=log_scales,
            quat=quat,
            sh_degree=int(args.sh_degree),
            binary=bool(args.binary),
        )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
