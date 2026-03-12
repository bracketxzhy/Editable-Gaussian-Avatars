# Editable Gaussian Avatars

## 1. What This Repository Is

This repository implements a monocular pipeline for building an editable Gaussian avatar from a single RGB video.

The actual execution chain is:

1. `preprocess_video.py`
2. `stage1_fit_smplerx.py`
3. `stage2_train_gaussian.py`
4. `render.py`

The short version:

- Stage 1 uses vendored `SMPLer-X` to estimate per-frame human body parameters and exports stage2-friendly assets.
- Stage 2 initializes Gaussians on the SMPL mesh sequence, binds them to the moving mesh, and optimizes them against RGB and silhouette supervision.
- Render reconstructs `real`, `fg_only`, and `deform` views from the trained checkpoint.

This is not a clean-room single-model system. It is a practical engineering pipeline built around:

- `SMPLer-X` for per-frame body estimation
- `MODNet` for person-mask supervision
- `gsplat` for Gaussian rendering

## 2. Project Goal

The repository aims to recover an editable avatar from monocular video with two practical properties:

- It should follow the body motion from a per-frame SMPL/SMPL-X mesh sequence.
- It should support appearance details that the body mesh cannot explain well, such as hair, clothing boundaries, and detached residual structure.

That second point is why stage2 contains inner/outer Gaussian logic instead of a single flat cloud.

## 3. Tasks Solved by the Repository

The repository solves these tasks:

- Extract frames from a monocular RGB video while preserving the original resolution.
- Estimate frame-wise body parameters and camera intrinsics using `SMPLer-X`.
- Build stage2 training assets:
  - `cameras.json`
  - `keypoints.json`
  - `person_masks.npz`
  - `smpl_params_sequence.npz`
  - `smpl_vertices_sequence.npz`
  - `smpl_joints_sequence.npz`
  - `smpl_skeleton_sequence.npz`
  - `sequence/frame_*.obj`
- Train mesh-bound 3D Gaussians from those assets.
- Render real-pose, foreground-only, and deform/standard-pose views from the checkpoint.
- Evaluate stage1 silhouette alignment quality.

## 4. Current Reality You Should Not Ignore

There is an important behavioral split in the current codebase:

- The codebase contains substantial dual-layer and outer-shell logic.
- But the default config `configs/stage2_gaussian.yaml` currently sets:

```yaml
stage2:
  training_mode: inner_only
```

That means the default stage2 command does **not** run a full dual-layer training loop by default.

If you blindly say you "reproduced the full pipeline" using the stock config, you did not. You reproduced the current default behavior, which is `inner_only`.

The PowerShell helper `scripts/auto_full_pipeline.ps1` also uses the same default config, so its name is more optimistic than accurate.

## 5. High-Level Architecture

### 5.1 Stage 0: Video Preprocess

Input:

- `data/videos/input.mp4`

Output:

- `data/frames/input/*.jpg`
- `data/frames/input/metadata.json`

Core file:

- `src/avatar_gs/cli/preprocess_video.py`

### 5.2 Stage 1: SMPLer-X Asset Export

Input:

- extracted frames
- `third_party/SMPLer-X`
- SMPLer-X pretrained checkpoint
- SMPL/SMPL-X body model files
- MODNet checkpoint

Output:

- `outputs/stage1/cameras.json`
- `outputs/stage1/keypoints.json`
- `outputs/stage1/person_masks.npz`
- `outputs/stage1/smpl_params_sequence.npz`
- `outputs/stage1/smpl_vertices_sequence.npz`
- `outputs/stage1/smpl_joints_sequence.npz`
- `outputs/stage1/smpl_skeleton_sequence.npz`
- `outputs/stage1/sequence/frame_*.obj`
- `outputs/stage1/final/run_summary.json`

Core file:

- `src/avatar_gs/cli/stage1_fit_smplerx.py`

### 5.3 Stage 2: Gaussian Training

Input:

- stage1 assets
- `configs/stage2_gaussian.yaml`

Output:

- `outputs/stage2/gaussians3d.pt`
- `outputs/stage2/gaussians3d_inner.pt`
- `outputs/stage2/supervision_cache.npz`
- `outputs/stage2/supervision_summary.json`
- `outputs/stage2/outer_quality_metrics.jsonl`
- `outputs/stage2/outer_quality_report.json`
- `outputs/stage2/outer_spawn_per_frame.json`

Core files:

- `src/avatar_gs/cli/stage2_train_gaussian.py`
- `src/avatar_gs/gaussian/train_loop.py`
- `src/avatar_gs/stage2/mesh_runtime.py`
- `src/avatar_gs/stage2/supervision.py`
- `src/avatar_gs/training/outer_layer_training.py`

### 5.4 Render

Input:

- stage2 checkpoint
- stage1 mesh / parameter assets

Output:

- `outputs/renders/real/...`
- `outputs/renders/fg_only/...`
- `outputs/renders/deform/...`

Core file:

- `src/avatar_gs/cli/render.py`

## 6. Directory Layout

```text
Editable Gaussian Avatars/
|- configs/
|- data/
|- evaluation/
|- outputs/
|- scripts/
|- src/
|  |- avatar_gs/
|  \- lib/
|- third_party/
|  |- MODNet/
|  \- SMPLer-X/
|- README.md
|- PROJECT_MAP.md
|- idea.md
\- requirements.txt
```

For a file-by-file map, see:

- `PROJECT_MAP.md`
- `idea.md`

## 7. Environment Setup

### 7.1 Recommended Python Environment

Use a dedicated environment. The vendored `SMPLer-X` stack is not tolerant of random package drift.

Example with conda:

```powershell
conda create -n ega python=3.10 -y
conda activate ega
python -m pip install --upgrade pip
```

### 7.2 Install PyTorch

Install a CUDA-matching PyTorch build first. Pick the command that matches your GPU and driver from the official PyTorch installation page.

Example pattern:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 7.3 Install Repository Python Dependencies

```powershell
pip install -r requirements.txt
```

### 7.4 Install `mmcv-full`

`SMPLer-X` requires `mmcv-full==1.7.1`, and this usually needs a wheel URL matched to your PyTorch/CUDA combination.

Example used by this repository's helper script:

```powershell
pip install mmcv-full==1.7.1
```

If that fails, use the OpenMMLab wheel index that matches your local CUDA/PyTorch versions.

### 7.5 Install Vendored `mmpose` Utilities

`SMPLer-X` also expects its vendored transformer utilities to be installed in editable mode:

```powershell
pip install -v -e third_party/SMPLer-X/main/transformer_utils
```

### 7.6 Set `PYTHONPATH`

Most commands assume:

```powershell
$env:PYTHONPATH = "src"
```

For `gsplat` extension caching on Windows, it is also useful to set:

```powershell
$env:TORCH_EXTENSIONS_DIR = "$PWD\\.torch_extensions"
```

## 8. External Assets Required

The repository does not become self-sufficient just because the folders are vendored. You still need the actual model files.

### 8.1 SMPLer-X Repository

Expected default path:

- `third_party/SMPLer-X`

### 8.2 SMPLer-X Pretrained Checkpoint

Stage1 checks for a pretrained checkpoint either under:

- `third_party/SMPLer-X/pretrained_models/`
- or somewhere under `assets/`

The code searches for names like:

- `smpler_x_*.pth.tar`
- `SMPLer-X*.pth.tar`
- `SMPLerX*.pth.tar`

### 8.3 MODNet Checkpoint

Expected default path:

- `third_party/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt`

### 8.4 SMPL / SMPL-X Body Model Files

These are expected under your assets/body-model layout consumed by `SMPLer-X` and `smplx`.

If these files are missing, stage1 will not be reproducible.

## 9. How To Reproduce

### 9.1 Prepare Input Video

Place your monocular RGB video at:

- `data/videos/input.mp4`

Or override it from the command line.

### 9.2 Step 1: Extract Frames

```powershell
$env:PYTHONPATH = "src"
python -m avatar_gs.cli.preprocess_video --config configs/video.yaml
```

Example with override:

```powershell
$env:PYTHONPATH = "src"
python -m avatar_gs.cli.preprocess_video --config configs/video.yaml --override video.path=data/videos/input.mp4
```

Expected outputs:

- `data/frames/input/*.jpg`
- `data/frames/input/metadata.json`
- `outputs/logs/preprocess_video.log`

### 9.3 Step 2: Run Stage1

```powershell
$env:PYTHONPATH = "src"
python -m avatar_gs.cli.stage1_fit_smplerx --project_root . --frames_dir data/frames/input --assets_dir assets --out_dir outputs/stage1 --smplerx_frame_indices -1 --single_person
```

Expected outputs:

- `outputs/stage1/cameras.json`
- `outputs/stage1/person_masks.npz`
- `outputs/stage1/smpl_params_sequence.npz`
- `outputs/stage1/sequence/frame_*.obj`

### 9.4 Optional: Evaluate Stage1 Alignment

```powershell
$env:PYTHONPATH = "src"
python evaluation/stage1_eval/run_stage1_eval.py --stage1_dir outputs/stage1 --frames_dir data/frames/input
```

Use this before stage2 if your stage1 silhouettes look suspicious. Otherwise you will end up blaming geometry for mask/alignment problems.

### 9.5 Step 3: Train Stage2

```powershell
$env:PYTHONPATH = "src"
$env:TORCH_EXTENSIONS_DIR = "$PWD\\.torch_extensions"
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
```

Expected outputs:

- `outputs/stage2/gaussians3d.pt`
- `outputs/stage2/supervision_cache.npz`
- `outputs/stage2/outer_quality_metrics.jsonl`

### 9.6 Step 4: Render

```powershell
$env:PYTHONPATH = "src"
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml
```

Expected outputs:

- `outputs/renders/real/...`
- `outputs/renders/fg_only/...`
- `outputs/renders/deform/...`

### 9.7 Single-Command Automation

If you want the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/auto_full_pipeline.ps1
```

Again: this script is only as honest as the config it calls. Right now that means `inner_only` unless you change the config.

## 10. Reproducibility Notes

To make runs more reproducible:

- keep `seed` fixed in configs
- avoid silently changing `training_mode`
- archive the resolved config files in `outputs/logs/`
- keep stage1 assets and stage2 supervision cache from the same config/data revision
- do not compare runs if one used cached supervision and the other recomputed it with different settings

The repository already writes resolved configs into `outputs/logs/`. Use them. If you do not archive the exact config, your "reproduction" claim is weak.

## 11. Main Configs

- `configs/video.yaml`
  - video extraction config
- `configs/stage1_smplerx.yaml`
  - stage1 path and export config
- `configs/stage2_gaussian.yaml`
  - main stage2 training/render config
- `configs/stage2_gaussian_firstframe.yaml`
  - first-frame experimental config

## 12. Main Outputs

### Stage1

- `outputs/stage1/cameras.json`
- `outputs/stage1/keypoints.json`
- `outputs/stage1/person_masks.npz`
- `outputs/stage1/smpl_params_sequence.npz`
- `outputs/stage1/smpl_vertices_sequence.npz`
- `outputs/stage1/sequence/frame_*.obj`

### Stage2

- `outputs/stage2/gaussians3d.pt`
- `outputs/stage2/gaussians3d_inner.pt`
- `outputs/stage2/supervision_cache.npz`
- `outputs/stage2/outer_quality_metrics.jsonl`
- `outputs/stage2/outer_quality_report.json`

### Render

- `outputs/renders/real/{combined,inner,outer}/*.png`
- `outputs/renders/fg_only/{combined,inner,outer}/*.png`
- `outputs/renders/deform/{combined,inner,outer}/*.png`

## 13. Common Failure Modes

### 13.1 "Stage1 ran, but stage2 looks wrong"

Usually not a useful diagnosis. Check:

- bad person masks
- poor stage1 silhouette alignment
- missing or inconsistent camera intrinsics
- wrong mesh coordinate-space assumptions

### 13.2 "Outer logic exists, but I see no outer behavior"

Check `training_mode`. The default config is currently `inner_only`.

### 13.3 "Render looks wrong but training seemed fine"

Render is not a thin viewer. It has its own logic for:

- deform mesh loading
- frame selection
- split-by-layer output
- alignment inheritance
- overlays

So a render issue is not automatically a training issue.

### 13.4 "Supervision cache made things reproducible"

Only if the cache actually matches the config and data revision. Otherwise you are just reusing stale supervision with confidence.

## 14. Limitations

- The repository depends heavily on external components (`SMPLer-X`, `MODNet`, `gsplat`).
- The default configuration does not currently reflect the full outer-shell story implied by the codebase.
- `train_loop.py` and `render.py` are large and carry too much responsibility.
- Several files are placeholders or compatibility shells rather than real production modules.

## 15. Practical Reading Order

If you are new to the repository, read in this order:

1. `README.md`
2. `idea.md`
3. `PROJECT_MAP.md`
4. `src/avatar_gs/cli/stage1_fit_smplerx.py`
5. `src/avatar_gs/cli/stage2_train_gaussian.py`
6. `src/avatar_gs/gaussian/train_loop.py`
7. `src/avatar_gs/cli/render.py`

## 16. Citation / Upstream Dependencies

This repository relies on major upstream components:

- `SMPLer-X`
- `MODNet`
- `gsplat`

If you use this repository in research, crediting only this wrapper code would be intellectually lazy. Cite the upstream work too.