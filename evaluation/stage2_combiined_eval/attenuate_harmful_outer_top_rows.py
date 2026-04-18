#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in THIS_FILE.parents if (p / "configs").is_dir() and (p / "src").is_dir())
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from avatar_gs.segmentation.fashn import get_fashn_label_maps, load_fashn_parser, predict_fashn_segmentation
from avatar_gs.segmentation.smpl_face_semantics import rasterize_visible_face_index_map
from avatar_gs.stage2 import load_stage2_supervision_cache
from evaluation.stage2_combiined_eval.run_stage2_combined_eval import merge_eval_checkpoint
from evaluation.stage2_eval.run_stage2_eval import EvalRenderer, clone_g, rgb_np, setup_logger, target_arrays


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def jfix(x):
    if isinstance(x, dict):
        return {str(k): jfix(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jfix(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return jfix(x.detach().cpu().numpy())
    return x


def abdomen_proxy_mask(part_mask: np.ndarray) -> np.ndarray:
    m = np.asarray(part_mask > 0, dtype=bool)
    ys, xs = np.where(m)
    if ys.size <= 0:
        return np.zeros_like(m, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = max(y1 - y0 + 1, 1)
    w = max(x1 - x0 + 1, 1)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    yn = (yy - y0).astype(np.float32) / max(h - 1, 1)
    xn = (xx - x0).astype(np.float32) / max(w - 1, 1)
    proxy = (xn >= 0.20) & (xn <= 0.82) & (yn >= 0.42) & (yn <= 0.88)
    return m & proxy


def project_points(xyz: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    xyz_cam = (xyz @ R.T) + t.view(1, 3)
    z = xyz_cam[:, 2]
    valid = z > 1.0e-6
    x = xyz_cam[:, 0] / z.clamp(min=1.0e-6)
    y = xyz_cam[:, 1] / z.clamp(min=1.0e-6)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v, valid


def parse_frames(value: str, all_frames: list[str]) -> list[str]:
    raw = str(value).strip().lower()
    if raw == "all":
        return list(all_frames)
    want = {x.strip() if x.strip().endswith(".jpg") else x.strip() + ".jpg" for x in str(value).replace(",", " ").split() if x.strip()}
    return [fn for fn in all_frames if fn in want]


def main() -> None:
    ap = argparse.ArgumentParser("Attenuate harmful outer top rows using all-frame abdomen-proxy projection attribution.")
    ap.add_argument("--project_root", default=str(PROJECT_ROOT))
    ap.add_argument("--stage1_dir", default="outputs/stage1")
    ap.add_argument("--stage2_dir", default="outputs/stage2")
    ap.add_argument("--inner_ckpt", default="outputs/stage2/gaussians3d.pt")
    ap.add_argument("--outer_ckpt", required=True)
    ap.add_argument("--supervision_cache", default="outputs/stage2/supervision_cache.npz")
    ap.add_argument("--smpl_face_semantics_npz", default="outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--frames", default="all")
    ap.add_argument("--alpha_threshold", type=float, default=0.03)
    ap.add_argument("--harmful_margin", type=float, default=0.006)
    ap.add_argument("--helpful_margin", type=float, default=0.006)
    ap.add_argument("--effect_min", type=float, default=0.0035)
    ap.add_argument("--sample_radius", type=int, default=3)
    ap.add_argument("--min_harm_score", type=float, default=0.002)
    ap.add_argument("--min_harm_hits", type=float, default=3.0)
    ap.add_argument("--min_net_ratio", type=float, default=0.24)
    ap.add_argument("--helpful_weight", type=float, default=0.70)
    ap.add_argument("--top_rows", type=int, default=36)
    ap.add_argument("--min_opacity_scale", type=float, default=0.55)
    ap.add_argument("--max_opacity_scale", type=float, default=0.82)
    ap.add_argument("--opacity_floor", type=float, default=0.01)
    args = ap.parse_args()

    project_root = resolve_path(Path.cwd(), args.project_root) if not Path(args.project_root).is_absolute() else Path(args.project_root).resolve()
    stage1_dir = resolve_path(project_root, args.stage1_dir)
    stage2_dir = resolve_path(project_root, args.stage2_dir)
    inner_ckpt = resolve_path(project_root, args.inner_ckpt)
    outer_ckpt = resolve_path(project_root, args.outer_ckpt)
    sup_path = resolve_path(project_root, args.supervision_cache)
    sem_path = resolve_path(project_root, args.smpl_face_semantics_npz)
    out_dir = resolve_path(project_root, args.out_dir)
    work_dir = resolve_path(project_root, args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "attenuate_harmful_outer_top_rows.log")

    z = np.load(str(sup_path), allow_pickle=True)
    all_frames = [str(x) for x in np.asarray(z["frame_names"]).reshape(-1).tolist()]
    frames = parse_frames(args.frames, all_frames)
    if not frames:
        raise RuntimeError("No frames selected.")

    merged_ckpt = work_dir / "combined_inner_existing_outer_d2.pt"
    merge_eval_checkpoint(
        inner_ckpt,
        outer_ckpt,
        merged_ckpt,
        work_dir / "combined_inner_existing_outer_d2_manifest.json",
        deactivate_outer_exposed=True,
        smpl_face_semantics_npz=sem_path,
    )
    sup = load_stage2_supervision_cache(cache_path=sup_path, frames=frames, logger=logger)
    if sup is None:
        raise RuntimeError(f"Cannot load supervision cache: {sup_path}")

    sem = np.load(str(sem_path), allow_pickle=True)
    raw_ids = np.asarray(sem["raw_label_ids"], dtype=np.int64).reshape(-1)
    raw_names = [str(x) for x in np.asarray(sem["raw_label_names"]).reshape(-1).tolist()]
    if "top" not in raw_names:
        raise RuntimeError("SMPL face semantics has no top label")
    top_raw_id = int(raw_names.index("top"))
    ids_to_labels, _ = get_fashn_label_maps(runtime_dir=None)
    top_fashn_id = next((int(rid) for rid, name in ids_to_labels.items() if str(name) == "top"), None)
    if top_fashn_id is None:
        raise RuntimeError("FASHN label maps have no top label")

    parser = load_fashn_parser(runtime_dir=None, model_dir=None, device="cuda" if torch.cuda.is_available() else "cpu")
    combined_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, "combined", float(args.alpha_threshold), logger)
    inner_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, "inner", float(args.alpha_threshold), logger)
    outer_renderer = EvalRenderer(project_root, stage1_dir, stage2_dir, merged_ckpt, frames, sup, "outer", 0.0, logger)

    outer = torch.load(str(outer_ckpt), map_location="cpu")
    outer_g = outer["gaussians"]
    shell = torch.as_tensor(outer_g["shell_layer"], dtype=torch.long).view(-1)
    prune = torch.as_tensor(outer_g["prune_state"], dtype=torch.long).view(-1)
    face = torch.as_tensor(outer_g["bind_face_idx"], dtype=torch.long).view(-1)
    valid_face_cpu = (face >= 0) & (face < int(raw_ids.shape[0]))
    face_raw = torch.full_like(face, -1)
    if bool(valid_face_cpu.any()):
        face_raw[valid_face_cpu] = torch.as_tensor(raw_ids, dtype=torch.long)[face[valid_face_cpu]]
    candidate_cpu = (shell == 1) & (prune == 0) & valid_face_cpu & (face_raw == int(top_raw_id))
    candidate_idx = torch.where(candidate_cpu.to(device=combined_renderer.device))[0]
    if int(candidate_idx.numel()) <= 0:
        raise RuntimeError("No active top outer rows found.")
    logger.info("candidate top outer rows=%d frames=%d", int(candidate_idx.numel()), len(frames))

    n_total = int(shell.shape[0])
    harm_sum = torch.zeros((n_total,), dtype=torch.float32, device=combined_renderer.device)
    help_sum = torch.zeros_like(harm_sum)
    active_sum = torch.zeros_like(harm_sum)
    harm_hits = torch.zeros_like(harm_sum)
    help_hits = torch.zeros_like(harm_sum)
    active_hits = torch.zeros_like(harm_sum)
    frames_harm = torch.zeros_like(harm_sum)

    k = max(1, 2 * int(args.sample_radius) + 1)
    kernel = np.ones((k, k), dtype=np.uint8)

    for i, frame_name in enumerate(frames):
        gt_u8, eval_mask = target_arrays(sup, "combined", i)
        gt = np.asarray(gt_u8, dtype=np.float32) / 255.0
        pred_c_t, _, _ = combined_renderer.render(i, frame_name)
        pred_i_t, _, _ = inner_renderer.render(i, frame_name)
        _, outer_alpha_t, _ = outer_renderer.render(i, frame_name)
        pred_c = rgb_np(pred_c_t)
        pred_i = rgb_np(pred_i_t)
        outer_alpha = np.asarray(outer_alpha_t.detach().cpu().squeeze(0).numpy(), dtype=np.float32).clip(0.0, 1.0)
        err_c = np.abs(pred_c - gt).mean(axis=2).astype(np.float32)
        err_i = np.abs(pred_i - gt).mean(axis=2).astype(np.float32)
        effect = np.abs(pred_c - pred_i).mean(axis=2).astype(np.float32)
        seg = predict_fashn_segmentation(np.asarray(gt_u8, dtype=np.uint8), parser=parser)

        R, t, K = combined_renderer.camera(frame_name)
        k_np = combined_renderer.frame_K(frame_name)
        verts_t, _ = combined_renderer.mesh_runtime.load_frame_mesh_verts(i, K_np=k_np, bind_cache_size=combined_renderer.bind_cache_size)
        verts_cam = (verts_t.detach().cpu().numpy().astype(np.float32) @ R.detach().cpu().numpy().astype(np.float32).T) + t.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
        faces_np = np.asarray(combined_renderer.mesh_runtime.context.bind_faces_np, dtype=np.int32)
        face_index_map, _ = rasterize_visible_face_index_map(
            verts_cam=verts_cam,
            faces=faces_np,
            K=K.detach().cpu().numpy().astype(np.float32),
            image_hw=(int(gt.shape[0]), int(gt.shape[1])),
        )
        valid_face_img = (face_index_map >= 0) & (face_index_map < int(raw_ids.shape[0]))
        part_mask = (np.asarray(seg, dtype=np.int32) == int(top_fashn_id)) & np.asarray(eval_mask, dtype=bool) & valid_face_img
        part_mask &= raw_ids[face_index_map.clip(min=0)] == int(top_raw_id)
        proxy = abdomen_proxy_mask(part_mask)
        if int(proxy.sum()) < 64:
            continue

        delta_h = np.maximum(err_c - err_i - float(args.harmful_margin), 0.0).astype(np.float32)
        delta_help = np.maximum(err_i - err_c - float(args.helpful_margin), 0.0).astype(np.float32)
        effect_gate = (effect > float(args.effect_min))
        alpha_gate = outer_alpha > float(args.alpha_threshold)
        harmful = proxy & alpha_gate & effect_gate & (delta_h > 0.0)
        helpful = proxy & alpha_gate & effect_gate & (delta_help > 0.0)
        active = proxy & alpha_gate & effect_gate
        harm_map = cv2.dilate((delta_h * effect * harmful.astype(np.float32)).astype(np.float32), kernel, iterations=1)
        help_map = cv2.dilate((delta_help * effect * helpful.astype(np.float32)).astype(np.float32), kernel, iterations=1)
        active_map = cv2.dilate((effect * active.astype(np.float32)).astype(np.float32), kernel, iterations=1)

        gd = clone_g(combined_renderer.g)
        gd, valid_rows = combined_renderer.apply_binding(i, frame_name, gd)
        u, v, zvalid = project_points(gd.xyz.index_select(0, candidate_idx), K, R, t)
        uu = torch.round(u).to(dtype=torch.long)
        vv = torch.round(v).to(dtype=torch.long)
        in_img = zvalid & (uu >= 0) & (uu < int(gt.shape[1])) & (vv >= 0) & (vv < int(gt.shape[0]))
        if valid_rows is not None:
            in_img = in_img & valid_rows.index_select(0, candidate_idx).to(device=combined_renderer.device)
        opacity_ok = torch.sigmoid(gd.opacity_logit.index_select(0, candidate_idx)).squeeze(-1) >= float(args.alpha_threshold)
        in_img = in_img & opacity_ok
        if not bool(in_img.any()):
            continue
        rows = candidate_idx[in_img]
        vv_np = vv[in_img].detach().cpu().numpy().astype(np.int64)
        uu_np = uu[in_img].detach().cpu().numpy().astype(np.int64)
        hv = torch.from_numpy(harm_map[vv_np, uu_np].astype(np.float32)).to(device=combined_renderer.device)
        hpv = torch.from_numpy(help_map[vv_np, uu_np].astype(np.float32)).to(device=combined_renderer.device)
        av = torch.from_numpy(active_map[vv_np, uu_np].astype(np.float32)).to(device=combined_renderer.device)
        harm_sum.index_add_(0, rows, hv)
        help_sum.index_add_(0, rows, hpv)
        active_sum.index_add_(0, rows, av)
        harm_hits.index_add_(0, rows, (hv > 0.0).to(dtype=torch.float32))
        help_hits.index_add_(0, rows, (hpv > 0.0).to(dtype=torch.float32))
        active_hits.index_add_(0, rows, (av > 0.0).to(dtype=torch.float32))
        frame_harm_rows = rows[hv > 0.0].unique()
        if int(frame_harm_rows.numel()) > 0:
            frames_harm.index_add_(0, frame_harm_rows, torch.ones_like(frame_harm_rows, dtype=torch.float32))
        logger.info("frame=%s proxy=%d harmful=%d helpful=%d projected=%d harm_rows=%d", frame_name, int(proxy.sum()), int(harmful.sum()), int(helpful.sum()), int(rows.numel()), int(frame_harm_rows.numel()))

    net = harm_sum - float(args.helpful_weight) * help_sum
    denom = (harm_sum + help_sum).clamp_min(1.0e-8)
    net_ratio = net / denom
    candidate = candidate_cpu.to(device=combined_renderer.device) & (harm_sum >= float(args.min_harm_score)) & (harm_hits >= float(args.min_harm_hits)) & (net_ratio >= float(args.min_net_ratio)) & (net > 0.0)
    cand_idx = torch.where(candidate)[0]
    if int(cand_idx.numel()) > 0:
        order = torch.argsort(net[cand_idx], descending=True)
        chosen_idx = cand_idx[order[: max(0, int(args.top_rows))]]
    else:
        chosen_idx = torch.empty((0,), dtype=torch.long, device=combined_renderer.device)
    chosen = torch.zeros_like(candidate, dtype=torch.bool)
    if int(chosen_idx.numel()) > 0:
        chosen[chosen_idx] = True

    opacity = torch.as_tensor(outer_g["opacity_logit"], dtype=torch.float32).clone().view(-1)
    chosen_cpu = chosen.detach().cpu()
    before = torch.sigmoid(opacity[chosen_cpu]).clone() if bool(chosen_cpu.any()) else torch.empty((0,), dtype=torch.float32)
    scales = torch.empty((0,), dtype=torch.float32)
    if bool(chosen_cpu.any()):
        chosen_net = net[chosen].detach().cpu()
        if int(chosen_net.numel()) > 0:
            strength = (chosen_net / max(float(chosen_net.max().item()), 1.0e-8)).clamp(0.0, 1.0)
            min_scale = float(args.min_opacity_scale)
            max_scale = float(args.max_opacity_scale)
            scales = max_scale - (max_scale - min_scale) * strength
            scales = scales.clamp(min=min(min_scale, max_scale), max=max(min_scale, max_scale))
            new_alpha = torch.clamp(before * scales.to(dtype=before.dtype), min=float(args.opacity_floor), max=0.999)
            opacity[chosen_cpu] = torch.log(new_alpha / (1.0 - new_alpha)).to(dtype=opacity.dtype)
    after = torch.sigmoid(opacity[chosen_cpu]).clone() if bool(chosen_cpu.any()) else torch.empty((0,), dtype=torch.float32)
    outer_g["opacity_logit"] = opacity.reshape_as(outer_g["opacity_logit"])

    top_rows = []
    for row in cand_idx[torch.argsort(net[cand_idx], descending=True)[:200]].detach().cpu().tolist() if int(cand_idx.numel()) > 0 else []:
        top_rows.append({
            "row": int(row),
            "face_id": int(face[int(row)].item()),
            "harm_sum": float(harm_sum[int(row)].detach().cpu().item()),
            "help_sum": float(help_sum[int(row)].detach().cpu().item()),
            "active_sum": float(active_sum[int(row)].detach().cpu().item()),
            "net": float(net[int(row)].detach().cpu().item()),
            "net_ratio": float(net_ratio[int(row)].detach().cpu().item()),
            "harm_hits": float(harm_hits[int(row)].detach().cpu().item()),
            "help_hits": float(help_hits[int(row)].detach().cpu().item()),
            "active_hits": float(active_hits[int(row)].detach().cpu().item()),
            "frames_harm": float(frames_harm[int(row)].detach().cpu().item()),
            "chosen": bool(chosen[int(row)].detach().cpu().item()),
        })

    manifest = {
        "source": str(outer_ckpt),
        "inner_ckpt": str(inner_ckpt),
        "parts": ["top"],
        "frames": frames,
        "candidate_rows": int(candidate_idx.numel()),
        "eligible_rows": int(candidate.sum().item()),
        "attenuated_rows": int(chosen.sum().item()),
        "alpha_threshold": float(args.alpha_threshold),
        "harmful_margin": float(args.harmful_margin),
        "helpful_margin": float(args.helpful_margin),
        "effect_min": float(args.effect_min),
        "sample_radius": int(args.sample_radius),
        "min_harm_score": float(args.min_harm_score),
        "min_harm_hits": float(args.min_harm_hits),
        "min_net_ratio": float(args.min_net_ratio),
        "helpful_weight": float(args.helpful_weight),
        "top_rows_requested": int(args.top_rows),
        "min_opacity_scale": float(args.min_opacity_scale),
        "max_opacity_scale": float(args.max_opacity_scale),
        "opacity_floor": float(args.opacity_floor),
        "opacity_change_mean_abs": float((after - before).abs().mean().item()) if int(before.numel()) > 0 else 0.0,
        "opacity_change_max_abs": float((after - before).abs().max().item()) if int(before.numel()) > 0 else 0.0,
        "opacity_scale_mean": float(scales.mean().item()) if int(scales.numel()) > 0 else 0.0,
        "selected_rows": [int(x) for x in torch.where(chosen)[0].detach().cpu().tolist()],
        "top_rows": top_rows,
    }
    outer["outer_harmful_top_row_attenuation"] = manifest
    torch.save(outer, str(out_dir / "gaussians3d.pt"))
    (out_dir / "harmful_top_row_attenuation_manifest.json").write_text(json.dumps(jfix(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jfix({k: v for k, v in manifest.items() if k not in {"frames", "top_rows"}}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
