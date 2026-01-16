# Project Map — Editable Gaussian Avatars (Monocular Video)

This document defines the purpose of **every directory and file** in the repository.
It is intended to be given directly to an AI so it can continue development without
re-asking structural questions.

---

## Overall Pipeline (One Sentence)

Monocular video → frame extraction → (optional PDR / depth) → SMPL fitting →
SMPL-structured Gaussian construction → differentiable Gaussian splatting rendering →
SMPL-parameter-based avatar editing.

---

## Top-Level Files

- `README.md`  
  Project entry. High-level description and how to run each stage.

- `requirements.txt`  
  Python dependencies required to run the pipeline.

---

## configs/ — All experiment parameters (NO hard-coded values)

- `configs/base.yaml`  
  Global configuration: seed, device, logging directory, default output paths.

- `configs/video.yaml`  
  Monocular video preprocessing config:

  - input video path
  - frame extraction step
  - resize settings
  - output frames directory

- `configs/stage1_smplfit.yaml`  
  Stage 1 configuration:

  - SMPL fitting parameters
  - optimization iterations
  - regularization weights
  - input frames directory
  - output SMPL params / mesh paths

- `configs/stage2_gaussian.yaml`  
  Stage 2 configuration:

  - structured Gaussian initialization rules
  - training hyperparameters
  - loss weights
  - number of iterations
  - checkpoint/output paths

- `configs/render.yaml`  
  Rendering configuration:

  - camera trajectory
  - output resolution
  - frame rate
  - render mode (reconstruction / novel view)

- `configs/edit.yaml`  
  Editing configuration:
  - custom SMPL pose / shape input
  - interpolation settings
  - joint-level edit options
  - output directory

---

## scripts/ — One-command execution helpers

- `scripts/run_all.sh`  
  Run the full pipeline: preprocess → stage1 → stage2 → render.

- `scripts/extract_frames.sh`  
  Only run video preprocessing (frame extraction).

- `scripts/stage1_fit_smpl.sh`  
  Only run Stage 1 (frames → SMPL parameters + mesh).

- `scripts/stage2_train_gaussian.sh`  
  Only run Stage 2 (SMPL → structured Gaussians → optimization).

- `scripts/render.sh`  
  Render images/videos from trained Gaussians.

- `scripts/edit.sh`  
  Render results from edited SMPL parameters.

---

## data/ — Input and intermediate data (not tracked by git)

- `data/videos/`  
  Raw monocular input videos (e.g., `input.mp4`).

- `data/frames/`  
  Extracted frames for each sequence:

  - image files (jpg/png)
  - `metadata.json` describing frame indices and video info

- `data/masks/`  
  Optional foreground/person masks (if background removal is used).

- `data/cameras/`  
  Camera parameters or pseudo camera trajectories
  (fixed camera, estimated motion, or default synthetic path).

---

## outputs/ — All generated results (not tracked by git)

- `outputs/stage1/`  
  Stage 1 outputs:

  - `smpl_params.npz`
  - `smpl_mesh.obj`
  - fitting logs

- `outputs/stage2/`  
  Stage 2 outputs:

  - `gaussians.pt`
  - training checkpoints
  - training metrics

- `outputs/renders/`  
  Rendered images/videos from trained Gaussians.

- `outputs/edits/`  
  Rendered images/videos after SMPL-based editing.

---

## src/avatar_gs/ — Core Python package

### Package Root

- `src/avatar_gs/__init__.py`  
  Marks `avatar_gs` as a Python package.

---

### cli/ — Command-line entry points (one file = one stage)

- `cli/run_all.py`  
  Python entry for the full pipeline.

- `cli/preprocess_video.py`  
  Monocular video preprocessing:
  video → frames + metadata.

- `cli/stage1_fit_smpl.py`  
  Stage 1 entry:
  frames / point cloud → SMPL parameters + SMPL-fitted mesh.

- `cli/stage2_train_gaussian.py`  
  Stage 2 entry:
  SMPL → structured Gaussians → optimization.

- `cli/render.py`  
  Rendering entry:
  Gaussians + cameras → images / videos.

- `cli/edit.py`  
  Editing entry:
  custom SMPL parameters → updated Gaussians → rendered output.

---

### data/ — Dataset and preprocessing utilities

- `data/video_reader.py`  
  Low-level video frame reader (OpenCV-based).

- `data/frame_dataset.py`  
  Dataset wrapper for extracted frames (and optional masks).

- `data/mask_utils.py`  
  Utilities for loading/resizing/applying masks.

- `data/camera_utils.py`  
  Camera parameter handling:
  loading, default monocular camera, synthetic trajectories.

---

### pdr/ — Monocular depth / point cloud reconstruction

- `pdr/monocular_pdr.py`  
  Interface for monocular depth or point cloud prediction.
  Can be a placeholder or replaced by a real model.

---

### smpl/ — SMPL model and fitting

- `smpl/smpl_model.py`  
  SMPL model wrapper:
  pose/shape → vertices, joints, skinning weights.

- `smpl/lbs.py`  
  Linear Blend Skinning and skeletal transformations (differentiable).

- `smpl/fit_from_video.py`  
  Fit SMPL parameters from video frames / point clouds.
  Outputs SMPL parameters and fitted mesh.

---

### gaussian/ — Core contribution: SMPL-Structured Gaussians

- `gaussian/structured_gaussian.py`  
  Definition of SMPL-structured Gaussian primitives:

  - Gaussian centers bound to SMPL vertices
  - Covariances defined in bone-local frames
  - Skinning-aware weights

- `gaussian/init_from_smpl.py`  
  Initialize Gaussians from an SMPL mesh
  (vertex sampling, initial covariance, color, opacity).

- `gaussian/losses.py`  
  Loss functions for Gaussian optimization
  (photometric, regularization, structure constraints).

- `gaussian/train_loop.py`  
  Training / optimization loop:
  forward render → loss → backward → checkpoint.

---

### renderer/ — Differentiable Gaussian splatting

- `renderer/gaussian_renderer.py`  
  Differentiable Gaussian splatting renderer interface.

- `renderer/camera.py`  
  Camera model and coordinate transformations
  (world / camera / projection).

---

### utils/ — Infrastructure (used everywhere)

- `utils/cfg.py`  
  YAML config loading, overriding, and saving resolved configs.

- `utils/io.py`  
  Common I/O utilities (json, npz, mesh, directory creation).

- `utils/seed.py`  
  Random seed control for reproducibility.

- `utils/log.py`  
  Logging setup (console + file).

---

## End-to-End File Flow (Strict Convention)

1. `preprocess_video.py`  
   → `data/frames/<seq>/*.jpg`  
   → `data/frames/<seq>/metadata.json`

2. `stage1_fit_smpl.py`  
   → `outputs/stage1/smpl_params.npz`  
   → `outputs/stage1/smpl_mesh.obj`

3. `stage2_train_gaussian.py`  
   → `outputs/stage2/gaussians.pt`  
   → `outputs/stage2/checkpoints/`

4. `render.py` / `edit.py`  
   → `outputs/renders/` or `outputs/edits/`

---

## Instruction to AI

Assume:

- `PYTHONPATH=src`
- All paths follow the conventions above
- Each stage reads outputs from the previous stage
- No file should redefine responsibilities already described here
