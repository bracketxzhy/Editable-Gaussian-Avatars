# PROJECT_MAP.md

## Repo Scope
- Project: `Editable Gaussian Avatars`
- Runtime: `PYTHONPATH=src`
- Main pipeline: `preprocess -> stage1 -> stage2 -> render`

## Stage2 Current Flow (Authoritative)
1. Load Stage1 assets
- mesh sequence / vertices sequence / person masks / camera intrinsics

2. Initialize Gaussians
- `gaussian/init_from_smpl.py`
- build inner+outer slots and binding metadata

3. Build per-frame binding data
- `gaussian/binding_layers/frame_builder.py`
- normals / tangents / face frames / curvature features

4. Per-iter training loop
- `gaussian/train_loop.py`
- pass-1 render -> per-gaussian visibility
- inner/outer per-frame binding update
- pass-2 weighted render -> photo losses + regularizers
- shell policy update (spawn/prune/softfreeze/hardprune)
- quality metrics logging + final quality report

5. Render outputs
- `cli/render.py`
- default six-views enabled
- outputs: `outputs/renders/real`, `outputs/renders/fg_only`, `outputs/renders/deform`

## Source Map (Stage2-critical)

### Entry / Orchestration
- `src/avatar_gs/cli/stage2_train_gaussian.py`
  - config parse, fail-fast key checks, pre-stage2 rigid align, call train loop
- `src/avatar_gs/cli/render.py`
  - real / fg_only / deform rendering and six-view export

### Gaussian Core
- `src/avatar_gs/gaussian/structured_gaussian.py`
  - `Gaussian3D` tensor schema and checkpoint serialization
- `src/avatar_gs/gaussian/train_loop.py`
  - stage2 core optimization and quality logging
- `src/avatar_gs/gaussian/shell_policy.py`
  - outer spawn/prune + inner freeze/prune policy functions
- `src/avatar_gs/gaussian/densify/stats.py`
  - densify statistics buffers (`vis_count`, `residual_ema`)
- `src/avatar_gs/gaussian/densify/rules.py`
  - densify candidate selection and stats update rules
- `src/avatar_gs/gaussian/losses.py`
  - basic gaussian-local loss helper
- `src/avatar_gs/gaussian/rigid_align.py`
  - rigid alignment utilities for SMPL/GT pre-alignment
- `src/avatar_gs/gaussian/init_from_smpl.py`
  - mesh loading and gaussian initialization

### Binding (Separated by Layer)
- `src/avatar_gs/gaussian/binding.py`
  - compatibility export layer
- `src/avatar_gs/gaussian/binding_layers/types.py`
  - binding dataclasses (`BindingModeConfig`, `BindingFrameData`)
- `src/avatar_gs/gaussian/binding_layers/frame_builder.py`
  - frame-level geometric frame construction
- `src/avatar_gs/gaussian/binding_layers/binding_update.py`
  - shared math + shared vertex target builder
- `src/avatar_gs/gaussian/binding_layers/inner_binding.py`
  - inner-only binding update path
- `src/avatar_gs/gaussian/binding_layers/outer_binding.py`
  - outer-only binding update path

### Training Helpers
- `src/avatar_gs/training/outer_layer_training.py`
  - trigger-ring construction (`build_outer_trigger_mask`)
- `src/avatar_gs/training/inner_layer_training.py`
  - strict inner supervision mask resolution
- `src/avatar_gs/training/frame_weighting.py`
  - frame residual EMA and frame weighting

### Visibility
- `src/avatar_gs/gaussian/visibility/gaussian_visibility.py`
  - projected visibility estimation
- `src/avatar_gs/gaussian/visibility/border_attenuation.py`
  - border attenuation map and sampling

## Config Contract (Current)
- Primary config: `configs/stage2_gaussian.yaml`
- Active section: `stage2`, `render`
- Deprecated keys are blocked in CLI (fail-fast), especially old reassign/vis-floor chain.

## Output Contract (Current)
- Stage2 checkpoint: `outputs/stage2/gaussians3d.pt`
- Stage2 quality logs:
  - `outputs/stage2/outer_quality_metrics.jsonl`
  - `outputs/stage2/outer_quality_report.json`
- Render outputs:
  - `outputs/renders/real`
  - `outputs/renders/fg_only`
  - `outputs/renders/deform`

## Canonical Commands
```powershell
$env:PYTHONPATH='src'
$env:TORCH_EXTENSIONS_DIR="$PWD\.torch_extensions"
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
```

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml --override render.deform_smplx_use_standard_pose=true
```
