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
- `evaluation/`
  - standalone evaluation scripts and output images (one-way dependency)
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

  - stage1 WHAM+PyMAF outputs (recommended unified stage1 output path)
  - includes runtime/intermediate/final artifacts produced by stage1 chain
  - stage2 gaussian checkpoints and training artifacts
  - rendered images/videos
  - edited render results
  - logs and resolved config snapshots

## 渲染输出要求

- 仅需生成以下两个输出文件夹（相对于项目根目录 `outputs/renders`）：
  - `fg_only`：前景无背景渲染（6 视图）
  - `t-pose`：T-pose 渲染（6 视图）

- 每个子文件夹中应包含六个视角的渲染图像（建议命名统一）：
  - `front.png`（正面）
  - `back.png`（后面）
  - `left.png`（左侧）
  - `right.png`（右侧）
  - `top.png`（俯视）
  - `bottom.png`（仰视）

- 渲染时要求：
  - 人体在图像中水平居中，盆骨位置接近图像中心；脚部不得接触图像边缘（使用适当的 `k_scale`/内参缩放以留出边界）。
  - 使用 `center_subject=true` 与 `center_subject_use_mesh=true` 保证以网格/绑定顶点为中心对齐。
  - 为避免恒模糊，请确保 stage2 的绑定（binding）与 stage1 导出的 mesh/params 对齐；若出现大量不对齐，先运行 stage1 导出一致的 `smpl_params_sequence.npz` / `smpl_vertices_sequence.npz`，或在训练前启用并完成 `bind_rigid_align` 的预对齐步骤。

## 推荐工作流程

1. 训练 Stage2（保证 `stage2.bind_to_mesh=true` 并且 `outputs/stage1` 存在数值 OBJ 与 `smpl_params_sequence.npz`）：

```powershell
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml --override stage2.stage1_dir=outputs/stage1 stage2.out_dir=outputs/stage2 stage2.init_mesh_obj=outputs/stage1/sequence/000756.obj
```

2. 训练完成后，生成六视图（脚本会为 `fg_only` 和 `t-pose` 各生成 6 张图并保存在 `outputs/renders` 下）：

```powershell
python scripts\generate_six_views.py --config configs/stage2_gaussian.yaml
```

3. 若训练结果恒模糊：检查并确保 `bind_rigid_align.pre_stage2=true` 并且 `outputs/stage1/person_masks.npz` 与 `smpl_params_sequence.npz` 对应帧名一致，或手动运行 stage1 使其导出正确的绑定资产后重训 stage2。

本文件是对渲染产物与训练对齐要求的简明说明，便于下游自动化与评估。
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
  - `stage1_align_check.py` (compat shim -> evaluation entry)

Evaluation entries:

- `evaluation/stage1_eval/run_stage1_eval.py`

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

set PYTHONPATH=src
python -m avatar_gs.cli.render --config configs/render.yaml

```

---

## Operational Rules

- Do not hard-code absolute paths in source files.
- Keep outputs under `outputs/`.
- Keep inputs under `data/`.
- Stage chaining should always be explicit via config/override paths.
- If unsure, prefer `stage1_fit_wham_pymaf` for stage1.


```
