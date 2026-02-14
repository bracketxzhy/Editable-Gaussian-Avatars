# PROJECT_MAP.md

## Repository

- Project: `Editable Gaussian Avatars`
- Root: this file sits at repo root.
- Primary runtime assumption: `PYTHONPATH=src`

## Canonical Pipeline (Current)

1. `preprocess_video`:
   - video -> frames
2. `stage1_fit_wham_pymaf`:
   - WHAM initialization + PyMAF refinement -> final SMPL/SMPL-X meshes and metadata
3. `stage2_train_gaussian`:
   - bind gaussians to SMPL mesh/sequence and optimize appearance
4. `render` / `edit`:
   - render `real`, `fg_only`, `t-pose`, novel views, and edited motions

Notes:
- Stage1 is WHAM+PyMAF only in the current repository.

---

## Top-Level Layout

- `assets/`
  - model files and checkpoints (SMPL/SMPL-X + WHAM/PyMAF related)
- `configs/`
  - all YAML configs used by CLI entries
- `data/`
  - inputs and preprocessed data
- `outputs/`
  - all generated results
- `scripts/`
  - convenience scripts
- `src/`
  - main Python package `avatar_gs`
- `third_party/`
  - external repos: `WHAM`, `PyMAF-X`
- `wham_pymaf_test/`
  - auxiliary workflow scripts for WHAM+PyMAF chain/stage2 bridge

---

## Config Files (Used)

- `configs/video.yaml`
  - frame extraction settings
- `configs/stage1_wham_pymaf.yaml`
  - stage1 WHAM+PyMAF settings
- `configs/stage2_gaussian.yaml`
  - stage2 gaussian training settings
- `configs/render.yaml`
  - rendering settings
- `configs/render_fg_only.yaml`
  - fg-only preview settings
- `configs/edit.yaml`
  - edit-time rendering settings

---

## Data Layout (Expected)

- `data/videos/input.mp4`
  - source monocular video
- `data/frames/input/*.jpg`
  - extracted frames (e.g. `000000.jpg` ...)
- `data/frames/input/metadata.json`
  - frame metadata from preprocessing
- `data/masks/`
  - optional masks
- `data/cameras/`
  - optional camera-related files

---

## Output Layout (Current Convention)

- `outputs/stage1/`
  - stage1 WHAM+PyMAF outputs (recommended unified stage1 output path)
  - includes runtime/intermediate/final artifacts produced by stage1 chain
- `outputs/stage2/`
  - stage2 gaussian checkpoints and training artifacts
- `outputs/renders/`
  - rendered images/videos
- `outputs/edits/`
  - edited render results
- `outputs/logs/`
  - logs and resolved config snapshots

---

## Source Package Map (`src/avatar_gs`)

### `cli/` (entry points)

- `preprocess_video.py`
- `stage1_fit_wham_pymaf.py` (default stage1)
- `stage2_train_gaussian.py`
- `render.py`
- `edit.py`
- `run_all.py`
- helper wrappers:
  - `_run_wham_on_frames.py`
  - `_run_pymaf_refine_frames.py`
  - `stage1_align_check.py`

### `data/`

- `video_reader.py`
- `frame_dataset.py`
- `keypoints_mediapipe.py`
- `mask_utils.py`
- `camera_utils.py`

### `smpl/`

- `smpl_model.py`
- `lbs.py`
- `fit_from_video.py` (compat bridge; legacy backend may be unavailable)

### `gaussian/`

- gaussian initialization, binding, losses, train loop

### `renderer/`

- gaussian renderer and camera projection utilities

### `utils/`

- cfg/io/log/seed and common helpers

---

## Third-Party Dependencies

- `third_party/WHAM`
  - WHAM codebase used for stage1 initialization
- `third_party/PyMAF-X`
  - PyMAF-X codebase used for frame refinement

Typical checkpoints are under:
- `assets/checkpoints/`

---

## Recommended Commands

### 1) Preprocess video to frames

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.preprocess_video --config configs/video.yaml --override video.path=data/videos/input.mp4 video.out_frames_dir=data/frames/input
```

### 2) Stage1 (WHAM + PyMAF) -> output to `outputs/stage1`

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.stage1_fit_wham_pymaf --project_root . --frames_dir data/frames/input --assets_dir assets --out_dir outputs/stage1 --pymaf_frame_indices -1 --single_person
```

### 3) Stage2 training

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
```

### 4) Render

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.render --config configs/render.yaml
```

---

## Operational Rules

- Do not hard-code absolute paths in source files.
- Keep outputs under `outputs/`.
- Keep inputs under `data/`.
- Stage chaining should always be explicit via config/override paths.
- If unsure, prefer `stage1_fit_wham_pymaf` for stage1.


