# Editable Gaussian Avatars - Handover Notes (English)

## 0. Mandatory Collaboration Rule

For **every training-related change** (Stage1, Stage2, render supervision, fitting priors, interface updates), `idea.md` must be updated in the same turn/commit.

Each update must include:

1. Training idea and motivation
2. Input/output contract changes
3. Interface/config changes
4. Validation method
5. Runnable commands (Windows CMD)
6. Risks and rollback

This file is the single handover source for other AI agents.

---

## 1. Project Goal

Build a monocular pipeline that:

1. Fits a stable, editable SMPL-X avatar from image sequence (`Stage1`)
2. Trains Gaussian appearance while keeping strong SMPL alignment (`Stage2`)
3. Supports reliable novel-view and pose editing without artifacts (no obvious drift, no body/background mismatch)

---

## 2. Current Pipeline (High-Level)

1. Stage1 (`avatar_gs.cli.stage1_fit_smpl`)
- Input: RGB frame sequence
- Extract MediaPipe keypoints and segmentation masks
- Fit SMPL-X sequence with soft physical priors + temporal smoothness
- Output canonical T-pose mesh, sequence meshes, motion params, skeleton sequence

2. Stage2 (`avatar_gs.cli.stage2_train_gaussian`)
- Initialize Gaussians from Stage1 mesh
- Bind foreground Gaussians to dynamic SMPL vertices
- Train photometric loss with mask-gated supervision
- Keep alignment for downstream editing/novel views

3. Render (`avatar_gs.cli.render`)
- Render final frame or selected frames
- Overlay SMPL skeleton from theta-driven sequence

---

## 3. I/O Contract

### 3.1 Main Inputs

- Frames: `data/frames/input`
- Stage1 config: `configs/stage1_smplfit.yaml`
- Stage2 config: `configs/stage2_gaussian.yaml`
- Render config: `configs/render_fg_only.yaml`

### 3.2 Stage1 Outputs (Required)

- `outputs/stage1/smpl_mesh_tpose.obj`
- `outputs/stage1/smpl_mesh_sequence/*.obj`
- `outputs/stage1/smpl_params_sequence.npz`
- `outputs/stage1/smpl_vertices_sequence.npz`
- `outputs/stage1/person_masks.npz`
- `outputs/stage1/smpl_skeleton_sequence.npz`

### 3.3 `smpl_skeleton_sequence.npz` fields

- `frame_names`
- `joints_cam_theta`
- `joint_parents`
- `model_type`
- `gender`
- `source=theta`

---

## 4. Interface and Config Contract

### 4.1 Stage1 (SMPL-X fitting)

Important keys:

- `stage1.init_translation_ls`
- `stage1.loss_schedule` (`balanced_sequence_v1`)
- `stage1.keypoint_smooth.*`
- `stage1.early_stop_enable`
- `stage1.early_stop_patience`
- `stage1.early_stop_min_delta`
- `stage1.early_stop_warmup_iters`

### 4.2 Stage2 (Gaussian training)

Important keys:

- `stage2.use_smpl_silhouette_mask`
- `stage2.smpl_mask_*`
- `stage2.use_person_mask`
- `stage2.person_masks_npz`
- `stage2.person_mask_dilate_px`
- `stage2.person_mask_erode_px`
- `stage2.person_mask_fuse_mode` (`intersect|union|replace`)
- `stage2.early_stop_enable`
- `stage2.early_stop_patience`
- `stage2.early_stop_min_delta`
- `stage2.early_stop_warmup_iters`

### 4.3 Render

- `render.smpl_skeleton_source` (`theta|joints`)
- `render.smpl_skeleton_npz`

---

## 5. Implemented Improvements (2026-02-11)

### 5.1 Alignment and body quality

1. Added CAR-style translation LS initialization in Stage1
2. Added balanced sequence loss schedule in Stage1
3. Added torso / hand-face keypoint weighting phases
4. Added extra foot anchors (heel + foot-index) for better lower-limb stability

### 5.2 Face stability for T-pose export

T-pose export now uses neutral face/eye/jaw/hand params (zeros) instead of temporal mean face pose, preventing severe face deformation.

### 5.3 Foreground/background robustness (not green-screen dependent)

Stage2 now supports Stage1 person masks:

- Load `outputs/stage1/person_masks.npz`
- Fuse person mask with existing alpha/kpt/smpl masks
- Default fusion mode: `intersect`

This makes supervision less dependent on green background assumptions and improves behavior on cluttered backgrounds.

### 5.4 Training speed improvements with accuracy guard

1. Stage1 early stopping (plateau detection)
2. Stage2 early stopping (plateau detection)

Goal: reduce wasted tail iterations while preserving converged quality.

### 5.5 Stage1 multi-frame body-model auto-selection prior (2026-02-11)

Motivation:
- Probe-only selection from a small frame subset can be unstable on ambiguous views.
- We need a conservative but less brittle route before deciding female/male vs neutral.

Training idea:
- Keep the original reprojection probe (`female|male|neutral`) as the primary evidence.
- Add a sequence-level visual prior from many sampled frames:
  - shoulder/hip ratio (keypoints)
  - hip/waist and chest/waist silhouette ratios (person mask)
  - head-top silhouette proxy (hair/head coverage)
- Fuse prior with probe scores using a small confidence-weighted shift.
- Keep conservative fallback: if evidence is weak, still select neutral.

I/O changes:
- No new mandatory files.
- `outputs/stage1/auto_gender_selection.json` now includes:
  - `scores_raw_probe`
  - `scores_fused`
  - `visual_prior` (metrics/signals/confidence)
  - effective thresholds after visual-confidence relaxation.

Interface changes (`configs/stage1_smplfit.yaml`):
- `stage1.auto_gender_visual_prior_enable`
- `stage1.auto_gender_visual_prior_weight`
- `stage1.auto_gender_visual_min_confidence`
- `stage1.auto_gender_visual_max_frames`
- `stage1.auto_gender_visual_use_masks`
- `stage1.auto_gender_visual_min_keypoint_conf`

Defaults chosen:
- Keep `auto_gender_confident_only: true`
- Use prior only as tie-breaker/soft evidence, not hard override.
- Increase probe frame subset from 32 to 48 for better stability.

### 2026-02-12 - Stage2/Render rigid-align wrapper for SMPL-vs-GT fit

1) Motivation
- Final frame overlay still had global offset (translation / yaw-pitch-roll mismatch) between SMPL silhouette and GT person.
- Need an extra alignment layer without changing SMPL intrinsic parameters (`pose/betas/trans` in Stage1 outputs).

2) Training idea changes
- Added a rigid wrapper (`R,t`) on top of per-frame SMPL vertices before Stage2 binding updates.
- Added two modes:
  - manual mode: use configured Euler angles + translation directly;
  - auto mode: optimize `(rx, ry, rz, tx, ty, tz)` by coordinate descent.
- Auto objective on sampled frames:
  - maximize SMPL-mask IoU with person mask;
  - penalize outside leakage (`score = IoU - outside_weight * outside_ratio`).
- The same rigid wrapper is reused at render time for mesh loading and skeleton overlay to keep train/render coordinate consistency.

3) I/O changes
- Input:
  - optional person mask NPZ for auto mode: `stage2.bind_rigid_align.person_masks_npz`.
- Output:
  - Stage2 checkpoint now stores `bind_rigid_align` metadata (enabled/auto/params and before-after metrics when auto mode runs).

4) Interface changes
- CLI:
  - `avatar_gs.cli.stage2_train_gaussian` forwards `stage2.bind_rigid_align` into train loop.
  - `avatar_gs.cli.render` reads `render.bind_rigid_align` override first, then falls back to checkpoint `bind_rigid_align`.
- YAML:
  - new block under `configs/stage2_gaussian.yaml`:
    - `stage2.bind_rigid_align.enable|auto`
    - `sample_frames`, `eval_downsample`, `outside_weight`
    - `rot_step_deg`, `trans_step`, `levels`, `iters_per_level`
    - `max_rot_deg`, `max_trans`
    - `person_masks_npz`
- Default behavior:
  - enabled + auto search by default in Stage2 config;
  - no SMPL parameter update, only extra global rigid transform before binding/render overlay.

5) Validation
- Smoke:
  - `python -m compileall src/avatar_gs/gaussian/rigid_align.py src/avatar_gs/gaussian/train_loop.py src/avatar_gs/cli/render.py src/avatar_gs/cli/stage2_train_gaussian.py`
- Metrics / visual checks:
  - verify Stage2 log prints `bind_rigid_align auto: score/iou/outside before->after`.
  - inspect render overlay for reduced background leakage and closer silhouette alignment.

6) Commands (Windows CMD)
~~~bat
conda activate ega-gpu
cd /d D:\project\Projects_by_zhy\Editable Gaussian Avatars
set PYTHONPATH=src

python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml --override render.overlay_smpl_skeleton=true
~~~

7) Risks / rollback
- Risk:
  - if person mask quality is poor, auto search may bias toward wrong rigid shift.
  - too large search bounds may overfit a sparse sampled subset.
- Rollback:
  - disable rigid wrapper with `stage2.bind_rigid_align.enable: false`.
  - or keep wrapper but disable auto search (`auto: false`) and set manual small offsets.

### 2026-02-12 - Move rigid-align search to pre-Stage2 gate

1) Motivation
- User concern: if rigid alignment starts only inside Stage2, early iterations may still absorb background leakage.
- We need to lock a better global `(R,t)` before optimization begins.

2) Training idea changes
- Added a pre-Stage2 rigid-align gate in `avatar_gs.cli.stage2_train_gaussian`.
- When `stage2.bind_rigid_align.enable=true`, `auto=true`, `pre_stage2=true`:
  - run coordinate-descent alignment before calling train loop;
  - freeze the searched params into manual mode (`auto=false`) for training.
- If pre-stage2 search cannot run (missing masks/faces/etc.), fallback remains in-train auto align.

3) I/O changes
- Input:
  - same as previous rigid-align auto mode (`person_masks_npz`, mesh sequence, frame mapping).
- Output:
  - new report file: `outputs/stage2/bind_rigid_align_pre_stage2.json`
  - resolved config includes `stage2.bind_rigid_align_effective`.

4) Interface changes
- CLI:
  - `stage2_train_gaussian` now triggers pre-stage2 rigid search automatically.
- YAML:
  - added `stage2.bind_rigid_align.pre_stage2` (default true).
- Default behavior:
  - run pre-stage2 search first, then train with fixed searched rigid params.

5) Validation
- Smoke:
  - `python -m compileall src/avatar_gs/cli/stage2_train_gaussian.py src/avatar_gs/gaussian/rigid_align.py`
- Metrics / visual checks:
  - check Stage2 log for `bind_rigid_align pre-stage2 auto: ...`
  - verify `outputs/stage2/bind_rigid_align_pre_stage2.json` exists.

6) Commands (Windows CMD)
~~~bat
conda activate ega-gpu
cd /d D:\project\Projects_by_zhy\Editable Gaussian Avatars
set PYTHONPATH=src

python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml --override render.overlay_smpl_skeleton=true
~~~

7) Risks / rollback
- Risk:
  - poor person masks can bias pre-stage2 alignment.
  - fixed-camera prior errors can affect pre-stage2 score landscape.
- Rollback:
  - `stage2.bind_rigid_align.pre_stage2: false` to disable pre-stage2 gate.
  - keep old behavior by letting in-train auto align handle search.

---

## 6. Current Recommended Defaults

### 6.1 Stage1

- Keep strong fitting (`fit_iters: 12000`) but enable early stop
- Keep `loss_schedule: balanced_sequence_v1`
- Keep soft physical priors (no hard pose-angle constraint)

### 6.2 Stage2

- Keep SMPL silhouette supervision
- Enable person mask fusion (`use_person_mask: true`)
- Enable early stop with warmup

---

## 7. Windows CMD - End-to-End Retrain

```bat
conda activate ega-gpu
cd /d D:\project\Projects_by_zhy\Editable Gaussian Avatars
set PYTHONPATH=src

if exist outputs\stage1 rmdir /s /q outputs\stage1
if exist outputs\stage2 rmdir /s /q outputs\stage2
if exist outputs\renders\fg_only rmdir /s /q outputs\renders\fg_only

python -m avatar_gs.cli.stage1_fit_smpl --config configs/stage1_smplfit.yaml
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml --override render.smpl_skeleton_source=theta
```

---

## 8. Acceptance Checklist

1. Required files exist after Stage1:
- `outputs/stage1/smpl_mesh_tpose.obj`
- `outputs/stage1/smpl_mesh_sequence/*.obj`
- `outputs/stage1/smpl_skeleton_sequence.npz`

2. Rendering checks:
- Skeleton lines include hand chains (from `joint_parents`)
- Reduced green/background contamination on legs
- Better SMPL-to-GT alignment for final frame

3. Editing safety checks:
- Novel views do not show obvious body/background mismatch
- Pose edits do not cause severe bind drift or body tearing


