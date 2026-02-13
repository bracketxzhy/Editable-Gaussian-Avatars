# WHAM + PyMAF-X Test

This folder contains a one-shot test pipeline:

1. WHAM initialization on your video frames
2. PyMAF(-X) refinement on selected frames
3. Export OBJ for the last frame
4. Render contour comparison on the GT last frame background:
   - GT human contour (yellow)
   - model contour (red)

## Run

From project root:

```bash
python wham_pymaf_test/run_wham_pymaf_test.py \
  --frames_dir data/frames/input \
  --assets_dir assets \
  --out_dir wham_pymaf_test/output \
  --pymaf_frame_indices -1 \
  --single_person
```

## Outputs

All outputs are written under `--out_dir`, including:

- `wham/` (WHAM outputs)
- `pymaf/selected_frames/` (PyMAF outputs)
- `final/last_frame_pose_local.obj`
- `final/last_frame_*_cam.obj` (camera/world metric mesh, recommended for stage2 binding)
- `final/last_frame_*.obj` (legacy local export for old scripts)
- `final/last_frame_coord_meta.json`
- `final/last_frame_gt_model_contours.png`
- `final/run_summary.json`

## One-Click Stage2 Bound Test

After `run_wham_pymaf_test.py` finishes, run:

```bash
python wham_pymaf_test/run_stage2_bound_from_wham_pymaf.py \
  --wham_pymaf_final_dir wham_pymaf_test/output_full_run/final \
  --stage1_dir outputs/stage1 \
  --stage1_bind_out_dir outputs/stage1_wham_pymaf_bind \
  --stage2_out_dir outputs/stage2_wham_pymaf_bind \
  --renders_out_dir outputs/renders_stage2_wham_pymaf_bind_3views \
  --iters 3000
```

This script will:
- build strict binding assets in camera/world coordinates
- run stage2 with `bind_to_mesh=true` on the last frame
- render `real / fg_only / t-pose`
- write summary files:
  - `outputs/stage1_wham_pymaf_bind/bind_asset_report.json`
  - `outputs/stage1_wham_pymaf_bind/stage2_chain_summary.json`

## Notes

- The script uses environment-based path adaptation to prefer models in `assets`.
- If required checkpoints are missing in `assets`, the script fails with explicit missing file hints.
