#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
from evaluation.stage2_combiined_eval.attenuate_harmful_outer_top_rows import abdomen_proxy_mask, jfix, parse_frames, project_points, resolve_path
from evaluation.stage2_combiined_eval.run_stage2_combined_eval import merge_eval_checkpoint
from evaluation.stage2_eval.run_stage2_eval import EvalRenderer, clone_g, rgb_np, setup_logger, target_arrays


def rgb_to_logit(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(rgb / (1.0 - rgb))


def main() -> None:
    ap = argparse.ArgumentParser("Rebalance active outer top RGB using automatic combined green/gray residual attribution.")
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
    ap.add_argument("--effect_min", type=float, default=0.0035)
    ap.add_argument("--sample_radius", type=int, default=3)
    ap.add_argument("--green_margin", type=float, default=0.006)
    ap.add_argument("--gray_margin", type=float, default=0.010)
    ap.add_argument("--dark_margin", type=float, default=0.012)
    ap.add_argument("--helpful_margin", type=float, default=0.004)
    ap.add_argument("--min_artifact_score", type=float, default=0.18)
    ap.add_argument("--min_artifact_hits", type=float, default=5.0)
    ap.add_argument("--min_artifact_ratio", type=float, default=0.28)
    ap.add_argument("--top_rows", type=int, default=96)
    ap.add_argument("--residual_gain", type=float, default=0.70)
    ap.add_argument("--max_rgb_delta", type=float, default=0.055)
    ap.add_argument("--green_cap_ratio", type=float, default=0.45, help="When green artifact dominates, cap green-channel increase relative to red/blue increase.")
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
    logger = setup_logger(out_dir / "rebalance_outer_top_green_residual.log")

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
    delta_sum = torch.zeros((n_total, 3), dtype=torch.float32, device=combined_renderer.device)
    artifact_sum = torch.zeros((n_total,), dtype=torch.float32, device=combined_renderer.device)
    protect_sum = torch.zeros_like(artifact_sum)
    green_sum = torch.zeros_like(artifact_sum)
    artifact_hits = torch.zeros_like(artifact_sum)
    frames_artifact = torch.zeros_like(artifact_sum)
    r = max(0, int(args.sample_radius))

    for i, frame_name in enumerate(frames):
        gt_u8, eval_mask = target_arrays(sup, "combined", i)
        gt = np.asarray(gt_u8, dtype=np.float32) / 255.0
        pred_c_t, _, _ = combined_renderer.render(i, frame_name)
        pred_i_t, _, _ = inner_renderer.render(i, frame_name)
        _, outer_alpha_t, _ = outer_renderer.render(i, frame_name)
        pred_c = rgb_np(pred_c_t)
        pred_i = rgb_np(pred_i_t)
        outer_alpha = np.asarray(outer_alpha_t.detach().cpu().squeeze(0).numpy(), dtype=np.float32).clip(0.0, 1.0)
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

        pred_luma = 0.299 * pred_c[:, :, 0] + 0.587 * pred_c[:, :, 1] + 0.114 * pred_c[:, :, 2]
        gt_luma = 0.299 * gt[:, :, 0] + 0.587 * gt[:, :, 1] + 0.114 * gt[:, :, 2]
        pred_sat = pred_c.max(axis=2) - pred_c.min(axis=2)
        gt_sat = gt.max(axis=2) - gt.min(axis=2)
        pred_green = pred_c[:, :, 1] - 0.5 * (pred_c[:, :, 0] + pred_c[:, :, 2])
        gt_green = gt[:, :, 1] - 0.5 * (gt[:, :, 0] + gt[:, :, 2])
        green_excess = np.maximum(pred_green - gt_green - float(args.green_margin), 0.0).astype(np.float32)
        gray_excess = np.maximum(gt_sat - pred_sat - float(args.gray_margin), 0.0).astype(np.float32)
        dark_excess = np.maximum(gt_luma - pred_luma - float(args.dark_margin), 0.0).astype(np.float32)
        effect = np.abs(pred_c - pred_i).mean(axis=2).astype(np.float32)
        err_c = np.abs(pred_c - gt).mean(axis=2).astype(np.float32)
        err_i = np.abs(pred_i - gt).mean(axis=2).astype(np.float32)
        active = proxy & (outer_alpha > float(args.alpha_threshold)) & (effect > float(args.effect_min))
        artifact_strength = (green_excess + 0.55 * gray_excess + 0.55 * dark_excess) * effect
        artifact_strength *= active.astype(np.float32)
        protect_strength = np.maximum(err_i - err_c - float(args.helpful_margin), 0.0).astype(np.float32) * effect * active.astype(np.float32)
        residual = (gt - pred_c).astype(np.float32)

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
        frame_art_rows = []
        for pos, row_t in enumerate(rows):
            y = int(vv_np[pos]); x = int(uu_np[pos])
            y0 = max(0, y - r); y1 = min(int(gt.shape[0]), y + r + 1)
            x0 = max(0, x - r); x1 = min(int(gt.shape[1]), x + r + 1)
            aw = artifact_strength[y0:y1, x0:x1]
            pw = protect_strength[y0:y1, x0:x1]
            a = float(aw.sum())
            p = float(pw.sum())
            row = int(row_t.detach().cpu().item())
            if a > 0.0:
                res = residual[y0:y1, x0:x1, :] * aw[:, :, None]
                delta_sum[row] += torch.from_numpy(res.sum(axis=(0, 1))).to(device=combined_renderer.device)
                artifact_sum[row] += a
                green_sum[row] += float((green_excess[y0:y1, x0:x1] * (aw > 0).astype(np.float32)).sum())
                artifact_hits[row] += 1.0
                frame_art_rows.append(row)
            if p > 0.0:
                protect_sum[row] += p
        if frame_art_rows:
            uniq = torch.as_tensor(sorted(set(frame_art_rows)), dtype=torch.long, device=combined_renderer.device)
            frames_artifact.index_add_(0, uniq, torch.ones_like(uniq, dtype=torch.float32))
        logger.info("frame=%s proxy=%d artifact_px=%d projected=%d artifact_rows=%d", frame_name, int(proxy.sum()), int((artifact_strength > 0).sum()), int(rows.numel()), len(set(frame_art_rows)))

    ratio = artifact_sum / (artifact_sum + protect_sum + 1.0e-8)
    eligible = candidate_cpu.to(device=combined_renderer.device) & (artifact_sum >= float(args.min_artifact_score)) & (artifact_hits >= float(args.min_artifact_hits)) & (ratio >= float(args.min_artifact_ratio))
    elig_idx = torch.where(eligible)[0]
    if int(elig_idx.numel()) > 0:
        order = torch.argsort(artifact_sum[elig_idx], descending=True)
        chosen_idx = elig_idx[order[: max(0, int(args.top_rows))]]
    else:
        chosen_idx = torch.empty((0,), dtype=torch.long, device=combined_renderer.device)
    chosen_cpu = chosen_idx.detach().cpu().to(dtype=torch.long)

    rgb_logit = torch.as_tensor(outer_g["rgb_logit"], dtype=torch.float32).clone()
    rgb_shape = rgb_logit.shape
    rgb = torch.sigmoid(rgb_logit.reshape(-1, 3))
    old_rgb = rgb[chosen_cpu].clone() if int(chosen_cpu.numel()) > 0 else torch.empty((0, 3), dtype=torch.float32)
    row_reports = []
    for row_t in chosen_cpu.tolist():
        row = int(row_t)
        mean_res = delta_sum[row].detach().cpu() / max(float(artifact_sum[row].detach().cpu().item()), 1.0e-8)
        delta = (float(args.residual_gain) * mean_res).clamp(-float(args.max_rgb_delta), float(args.max_rgb_delta))
        green_dominates = float(green_sum[row].detach().cpu().item()) > 0.35 * float(artifact_sum[row].detach().cpu().item())
        if green_dominates:
            rb_pos = max(float(delta[0].item()), 0.0) + max(float(delta[2].item()), 0.0)
            green_cap = float(args.green_cap_ratio) * 0.5 * rb_pos
            if float(delta[1].item()) > green_cap:
                delta[1] = green_cap
        before = rgb[row].clone()
        rgb[row] = (rgb[row] + delta.to(dtype=rgb.dtype)).clamp(1.0e-4, 1.0 - 1.0e-4)
        row_reports.append({
            "row": row,
            "face_id": int(face[row].item()),
            "artifact_sum": float(artifact_sum[row].detach().cpu().item()),
            "protect_sum": float(protect_sum[row].detach().cpu().item()),
            "artifact_ratio": float(ratio[row].detach().cpu().item()),
            "artifact_hits": float(artifact_hits[row].detach().cpu().item()),
            "frames_artifact": float(frames_artifact[row].detach().cpu().item()),
            "mean_residual": [float(x) for x in mean_res.tolist()],
            "rgb_before": [float(x) for x in before.detach().cpu().tolist()],
            "rgb_after": [float(x) for x in rgb[row].detach().cpu().tolist()],
        })
    new_rgb = rgb_to_logit(rgb).reshape(rgb_shape).to(dtype=rgb_logit.dtype)
    outer_g["rgb_logit"] = new_rgb
    new_sel_rgb = rgb[chosen_cpu].clone() if int(chosen_cpu.numel()) > 0 else torch.empty((0, 3), dtype=torch.float32)

    manifest = {
        "source": str(outer_ckpt),
        "inner_ckpt": str(inner_ckpt),
        "parts": ["top"],
        "frames": frames,
        "candidate_rows": int(candidate_idx.numel()),
        "eligible_rows": int(eligible.sum().detach().cpu().item()),
        "changed_rows": int(chosen_cpu.numel()),
        "alpha_threshold": float(args.alpha_threshold),
        "effect_min": float(args.effect_min),
        "sample_radius": int(args.sample_radius),
        "green_margin": float(args.green_margin),
        "gray_margin": float(args.gray_margin),
        "dark_margin": float(args.dark_margin),
        "helpful_margin": float(args.helpful_margin),
        "min_artifact_score": float(args.min_artifact_score),
        "min_artifact_hits": float(args.min_artifact_hits),
        "min_artifact_ratio": float(args.min_artifact_ratio),
        "top_rows_requested": int(args.top_rows),
        "residual_gain": float(args.residual_gain),
        "max_rgb_delta": float(args.max_rgb_delta),
        "green_cap_ratio": float(args.green_cap_ratio),
        "changed_rgb_mean_abs": float((new_sel_rgb - old_rgb).abs().mean().item()) if int(chosen_cpu.numel()) > 0 else 0.0,
        "changed_rgb_max_abs": float((new_sel_rgb - old_rgb).abs().max().item()) if int(chosen_cpu.numel()) > 0 else 0.0,
        "selected_rows": [int(x) for x in chosen_cpu.tolist()],
        "top_rows": row_reports[:200],
    }
    outer["outer_top_green_residual_rebalance"] = manifest
    torch.save(outer, str(out_dir / "gaussians3d.pt"))
    (out_dir / "top_green_residual_rebalance_manifest.json").write_text(json.dumps(jfix(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jfix({k: v for k, v in manifest.items() if k not in {"frames", "top_rows"}}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
