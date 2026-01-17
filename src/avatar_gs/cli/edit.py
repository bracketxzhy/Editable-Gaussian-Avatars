from __future__ import annotations

from pathlib import Path
import inspect
import numpy as np
import torch
import cv2

from avatar_gs.utils.cfg import load_yaml, apply_overrides, parse_args_base
from avatar_gs.utils.io import ensure_dir, read_json
from avatar_gs.utils.log import setup_logger

from avatar_gs.gaussian.init_from_smpl import init_gaussians_from_mesh_3d
from avatar_gs.renderer.gaussian_renderer import render_gaussians_3d


def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _normalize_K(K_like, H: int, W: int) -> np.ndarray:
    """
    Convert various intrinsics formats into a 3x3 K matrix.

    Supported:
      - 3x3 matrix
      - flat length-9
      - scalar focal length -> assumes fx=fy=f, cx=W/2, cy=H/2
      - length-4 [fx, fy, cx, cy]
      - dict with fx/fy/cx/cy (or focal variants)
    """
    if isinstance(K_like, dict):
        fx = K_like.get("fx", K_like.get("focal_x", K_like.get("f", None)))
        fy = K_like.get("fy", K_like.get("focal_y", fx))
        cx = K_like.get("cx", K_like.get("ppx", W * 0.5))
        cy = K_like.get("cy", K_like.get("ppy", H * 0.5))
        if fx is None:
            raise ValueError(f"Cannot normalize K from dict: {K_like}")
        return np.array([[float(fx), 0.0, float(cx)],
                         [0.0, float(fy), float(cy)],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    arr = np.asarray(K_like, dtype=np.float32)

    if arr.ndim == 0:
        f = float(arr)
        return np.array([[f, 0.0, W * 0.5],
                         [0.0, f, H * 0.5],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    if arr.shape == (3, 3):
        return arr

    if arr.size == 9:
        return arr.reshape(3, 3)

    if arr.size == 4:
        fx, fy, cx, cy = [float(x) for x in arr.reshape(-1)]
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    raise ValueError(f"Unrecognized K format/shape: shape={arr.shape}, size={arr.size}, value={K_like}")


def _pick_cam0(cam: dict, H: int, W: int):
    """
    Pick first camera and derive (K_3x3, R, t) in world->camera convention.
    Supports cameras.json variants:
      - K: [N][3][3] or [3][3] or scalar/flat/4-vec/dict
      - R/t lists (optional)
      - w2c list 4x4 (optional)
      - c2w list 4x4 (optional)
    """
    def first(v):
        if isinstance(v, list):
            return v[0]
        return v

    K0 = cam.get("K", None)
    if K0 is None:
        for k in ("intrinsics", "K_matrix", "K0"):
            if k in cam:
                K0 = cam[k]
                break
    if K0 is None:
        raise KeyError("cameras.json missing intrinsics (expected key 'K' or equivalent)")
    K0 = first(K0)
    K3 = _normalize_K(K0, H=H, W=W)

    # Preferred: explicit R/t
    R0 = cam.get("R", None)
    t0 = cam.get("t", None)
    if R0 is not None and t0 is not None:
        R = np.asarray(first(R0), dtype=np.float32)
        t = np.asarray(first(t0), dtype=np.float32)
        return K3, R, t

    # w2c 4x4
    w2c = cam.get("w2c", None) or cam.get("world_to_camera", None)
    if w2c is not None:
        M = np.asarray(first(w2c), dtype=np.float32)
        return K3, M[:3, :3], M[:3, 3]

    # c2w 4x4 -> invert
    c2w = cam.get("c2w", None) or cam.get("camera_to_world", None)
    if c2w is not None:
        M = np.asarray(first(c2w), dtype=np.float32)
        W2C = np.linalg.inv(M)
        return K3, W2C[:3, :3], W2C[:3, 3]

    raise KeyError("cameras.json missing extrinsics: expected (R,t) or w2c/c2w 4x4")


def _call_init_gaussians_from_mesh_3d(
    mesh: Path,
    cam: dict,
    n_keep: int,
    seed: int,
):
    """
    Robustly call init_gaussians_from_mesh_3d by matching its actual signature.
    This avoids breakage when parameter names differ across forks/versions.
    """
    sig = inspect.signature(init_gaussians_from_mesh_3d)
    params = sig.parameters

    pool = {
        # mesh
        "mesh_obj": mesh,
        "mesh_path": mesh,
        "obj_path": mesh,
        "obj": mesh,
        "path": mesh,

        # camera info (some impls accept these, some ignore them)
        "K_list": cam.get("K", None),
        "Ks": cam.get("K", None),
        "K": cam.get("K", None),

        "t_list": cam.get("t", None),
        "ts": cam.get("t", None),
        "t": cam.get("t", None),

        "R_list": cam.get("R", None),
        "Rs": cam.get("R", None),
        "R": cam.get("R", None),

        "w2c": cam.get("w2c", None) or cam.get("world_to_camera", None),
        "w2c_list": cam.get("w2c", None) or cam.get("world_to_camera", None),

        "c2w": cam.get("c2w", None) or cam.get("camera_to_world", None),
        "c2w_list": cam.get("c2w", None) or cam.get("camera_to_world", None),

        # init controls
        "n_keep": n_keep,
        "num_keep": n_keep,
        "n_points": n_keep,
        "num_points": n_keep,
        "N": n_keep,

        "seed": seed,
        "rng_seed": seed,
        "random_seed": seed,
    }

    kwargs = {}
    for name in params.keys():
        if name in pool and pool[name] is not None:
            kwargs[name] = pool[name]

    # Try keyword call first; then positional fallbacks
    try:
        return init_gaussians_from_mesh_3d(**kwargs)
    except TypeError:
        try:
            return init_gaussians_from_mesh_3d(mesh, **kwargs)
        except TypeError:
            return init_gaussians_from_mesh_3d(mesh)


def main():
    parser = parse_args_base("Edit SMPL params and render edited gaussian")
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)

    root = Path(cfg.get("project_root", ".")).resolve()
    out_dir = ensure_dir(root / "outputs" / "edits")
    logger = setup_logger("edit", None)

    # stage1
    stage1 = root / "outputs" / "stage1"
    cam = read_json(stage1 / "cameras.json")

    # image size (prefer cfg, otherwise defaults)
    H = int(cfg.get("H", cfg.get("height", 480)))
    W = int(cfg.get("W", cfg.get("width", 640)))

    # load & edit SMPL params (NOTE: this script does NOT regenerate mesh from pose)
    smpl = np.load(stage1 / "smpl_params.npz")
    pose = smpl["pose"].copy()
    if pose.size > 3:
        pose[3] += 0.5  # dummy edit; does not affect mesh unless you regenerate it

    # mesh (still using stage1 mesh as-is)
    mesh = stage1 / "smpl_mesh.obj"

    # init gaussians
    n_keep = int(cfg.get("n_keep", 600))
    seed = int(cfg.get("seed", 0))
    gauss = _call_init_gaussians_from_mesh_3d(mesh=mesh, cam=cam, n_keep=n_keep, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gauss = gauss.to(device)

    # camera 0
    K_np, R_np, t_np = _pick_cam0(cam, H=H, W=W)
    K = _to_tensor(K_np, device=device, dtype=torch.float32)
    R = _to_tensor(R_np, device=device, dtype=torch.float32)
    t = _to_tensor(t_np, device=device, dtype=torch.float32)

    img = render_gaussians_3d(gauss, K, R, t, H, W).detach().clamp(0, 1)  # [3,H,W]
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")       # [H,W,3] RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out = out_dir / "edit_000000.png"
    cv2.imwrite(str(out), img)
    logger.info(f"Saved edited render: {out}")


if __name__ == "__main__":
    main()
