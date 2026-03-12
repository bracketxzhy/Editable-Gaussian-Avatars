# PROJECT_MAP.md

## 说明边界

- 这份地图覆盖“本仓库负责人应维护的文件”：顶层脚本、配置、`src/avatar_gs/**`、`src/lib/**`、`scripts/**`、`evaluation/stage1_eval/**`。
- `third_party/**` 与 `outputs/**` 不是同权手写源码：前者是外部仓库，后者是运行产物。它们会写清边界、入口和数据契约，但不会假装成和自研源码同级的逐文件维护对象。
- 下面每个文件都按四件事写：职责、接口、输入、输出/下游。

## 顶层文件

### `.codex_mmcv_build.cmd`
- 职责：本地环境辅助脚本，进入 `ega-gpu` conda 环境并安装 `mmcv-full==1.7.1`。
- 接口：命令行脚本，无 Python API。
- 输入：当前仓库环境。
- 输出/下游：生成 `mmcv_build.log`，供本地依赖准备使用。

### `.codex_smoke_stage1.cmd`
- 职责：stage1 烟雾测试，先做 `py_compile`，再跑极小规模 stage1。
- 接口：命令行脚本。
- 输入：当前源码、配置、stage1 依赖。
- 输出/下游：验证 stage1 主链是否可启动。

### `requirements.txt`
- 职责：声明最小 Python 依赖。
- 接口：`pip install -r requirements.txt`。
- 输入：无。
- 输出/下游：提供运行时依赖，包括 `torch`、`opencv-python`、`scipy`、`mediapipe`、`smplx` 等。

### `idea.md`
- 职责：记录当前逻辑真相、分叉点、风险和维护建议。
- 接口：文档，无运行接口。
- 输入：代码现实。
- 输出/下游：给负责人和维护者提供判断基线。

### `PROJECT_MAP.md`
- 职责：记录文件级职责、接口和数据流地图。
- 接口：文档，无运行接口。
- 输入：代码现实。
- 输出/下游：给维护者快速定位系统边界和上下游。

### `outer`
- 职责：零字节残留文件，没有业务意义。
- 接口：无。
- 输入：无。
- 输出/下游：只会制造困惑。

### `mmcv_build.log`
- 职责：`mmcv` 安装日志。
- 接口：日志文件。
- 输入：`.codex_mmcv_build.cmd`。
- 输出/下游：本地环境排障。

## 配置文件

### `configs/video.yaml`
- 职责：定义预处理视频抽帧参数。
- 接口：被 `avatar_gs.cli.preprocess_video` 读取。
- 输入：视频路径、输出目录、抽帧步长、JPEG 质量。
- 输出/下游：控制 `data/frames/input/*.jpg` 和 `metadata.json` 的生成。

### `configs/stage1_smplerx.yaml`
- 职责：定义 stage1 路径、SMPLer-X 仓库、资产目录、输出目录。
- 接口：被 `avatar_gs.cli.stage1_fit_smplerx` 读取。
- 输入：帧目录、资产目录、SMPLer-X 目录、输出目录。
- 输出/下游：控制 stage1 导出资产位置。

### `configs/stage2_gaussian.yaml`
- 职责：stage2 主配置，覆盖帧选择、绑定、监督、outer 策略、渲染行为。
- 接口：被 `avatar_gs.cli.stage2_train_gaussian` 和 `avatar_gs.cli.render` 读取。
- 输入：stage1 输出目录、训练参数、监督参数、render 参数。
- 输出/下游：决定 stage2 默认行为。当前关键事实是默认 `training_mode: inner_only`。

### `configs/stage2_gaussian_firstframe.yaml`
- 职责：单帧快速实验配置。
- 接口：同上。
- 输入：与主配置同类，但偏单帧实验。
- 输出/下游：用于 first-frame 验证，不应误当主训练配置。

## 脚本与评估入口

### `scripts/auto_full_pipeline.ps1`
- 职责：等待已有任务结束后自动串行执行 stage1 -> stage2 -> render，并做简单轮次比较。
- 接口：PowerShell 脚本；主要函数 `Get-AvatarGsRuns`、`Log`。
- 输入：仓库路径、`configs/stage2_gaussian.yaml`、stage1/stage2/render 命令。
- 输出/下游：写 `outputs/logs/auto_full_pipeline.log`，保留分数最优的 `gaussians3d.pt` 和 `fg_only/000000.png`。
- 数据流：调用 `stage1_fit_smplerx` 生成 stage1 资产，调用 `stage2_train_gaussian` 训练，调用 `render` 输出图像，再用内联 Python 对结果做简评。
- 风险：脚本名叫 full，但当前默认配置实际上是 `inner_only`。

### `scripts/export_gaussians_to_ply.py`
- 职责：把 stage2 checkpoint 导出成点云风格或 3DGS 风格 PLY。
- 接口：`main()`；辅助函数 `_load_ckpt`、`_resolve_mesh_index`、`_write_ply_points`、`_write_ply_3dgs`。
- 输入：`outputs/stage2/gaussians3d.pt`、可选 mesh 索引和导出模式。
- 输出/下游：PLY 文件，供 Blender 或外部查看器调试。

### `evaluation/stage1_eval/person_contour.py`
- 职责：stage1 评估的人体 mask/contour 工具层。
- 接口：`load_person_mask_from_npz`、`estimate_person_mask_modnet_eval`、`extract_person_contours`。
- 输入：图片、`person_masks.npz`、MODNet。
- 输出/下游：干净的人体 mask 与 contour，供 `run_stage1_eval.py` 使用。

### `evaluation/stage1_eval/run_stage1_eval.py`
- 职责：比较 stage1 SMPL 投影轮廓与 GT/person mask 的对齐质量。
- 接口：`main()`、`_evaluate_one_frame()`、`_resolve_pick_index()`。
- 输入：stage1 `cameras.json`、`smpl_vertices_sequence.npz`/OBJ、人物 mask/contour。
- 输出/下游：单帧或多帧评估 JSON 与可视化图。

### `evaluation/stage1_eval/stage1_align_check.py`
- 职责：对 `run_stage1_eval.py` 的兼容包装入口。
- 接口：`main()`。
- 输入：命令行参数。
- 输出/下游：转发到 stage1 评估流程。

## `src/avatar_gs` 包根与导出层

### `src/avatar_gs/__init__.py`
- 职责：包占位。
- 接口：无。
- 输入：无。
- 输出/下游：仅让 `avatar_gs` 成为可导入包。

### `src/avatar_gs/alignment/__init__.py`
- 职责：导出 alignment 相关符号。
- 接口：包导出层。
- 输入：内部模块。
- 输出/下游：给外部更稳定的 import 路径。

### `src/avatar_gs/deform/__init__.py`
- 职责：导出 deform 顶点加载逻辑。
- 接口：包导出层。
- 输入：`smplx_param_source.py`。
- 输出/下游：给 render 或其他调用方使用。

### `src/avatar_gs/io/__init__.py`
- 职责：导出 `export_obj`。
- 接口：包导出层。
- 输入：`mesh_export.py`。
- 输出/下游：供 stage1 OBJ 导出使用。

### `src/avatar_gs/losses/__init__.py`
- 职责：导出 robust loss 和 regularizer。
- 接口：包导出层。
- 输入：`robust.py`、`regularizers.py`。
- 输出/下游：给训练循环使用。

### `src/avatar_gs/segmentation/__init__.py`
- 职责：导出 MODNet 推理接口。
- 接口：包导出层。
- 输入：`modnet.py`。
- 输出/下游：给 stage1/stage2/eval 统一调用。

### `src/avatar_gs/stage2/__init__.py`
- 职责：导出 stage2 核心配置、监督、mesh runtime、reporting、rigid align 符号。
- 接口：包导出层。
- 输入：`config.py`、`mesh_runtime.py`、`supervision.py`、`reporting.py`、`rigid_align.py`。
- 输出/下游：给 CLI 和训练主流程使用。

### `src/avatar_gs/training/__init__.py`
- 职责：导出 frame weighting、inner/outer training helper。
- 接口：包导出层。
- 输入：`frame_weighting.py`、`inner_layer_training.py`、`outer_layer_training.py`。
- 输出/下游：给训练循环使用。

## CLI 入口

### `src/avatar_gs/cli/run_all.py`
- 职责：最朴素的串行 orchestrator。
- 接口：`run(cmd)`、`main()`。
- 输入：固定的 preprocess、stage1、stage2、render 命令。
- 输出/下游：依次触发整个主流程；只适合简单串行调度。

### `src/avatar_gs/cli/preprocess_video.py`
- 职责：按原始分辨率抽帧并记录元数据。
- 接口：`main()`。
- 输入：`configs/video.yaml` 或命令行覆盖、输入视频。
- 输出/下游：`data/frames/input/*.jpg`、`metadata.json`、resolved yaml；下游供 stage1 读取。

### `src/avatar_gs/cli/_run_smplerx_refine_frames.py`
- 职责：为 stage1 组织 SMPLer-X 运行目录并只跑指定帧。
- 接口：`main()`；辅助函数 `_list_frames`、`_parse_indices`、`_safe_link_or_copy`、`_resolve_model_name`、`_ensure_pretrained_model`。
- 输入：抽帧目录、选帧索引、SMPLer-X 仓库、预训练模型。
- 输出/下游：向 `third_party/SMPLer-X/main/inference.py` 提供准备好的输入目录和模型路径。

### `src/avatar_gs/cli/stage1_fit_smplerx.py`
- 职责：stage1 主执行器，负责准备、运行、重建、导出。
- 接口：`main()`。
- 主要内部接口：
  - 运行准备：`_resolve_smplerx_ckpt`、`_prepare_smplerx_runtime_data`、`_check_stage1_requirements`
  - mask 估计：`_estimate_person_mask_chromakey`、`_estimate_person_mask_modnet_stage1`、`_mask_score`
  - 几何投影：`_camera_intrinsics_from_record`、`_project_verts_to_uv`、`_rasterize_mesh_mask`
  - 参数重建：`_load_smplerx_npz_params`、`_rebuild_smplx_from_params`
  - 记录整理：`_build_smplerx_frame_records`
  - 资产导出：`_write_stage1_sequence_assets_full`
- 输入：帧图像、SMPLer-X 输出、资产目录、模型权重、可选 MODNet。
- 输出/下游：
  - `cameras.json`
  - `keypoints.json`
  - `person_masks.npz`
  - `smpl_params_sequence.npz`
  - `smpl_vertices_sequence.npz`
  - `smpl_joints_sequence.npz`
  - `smpl_skeleton_sequence.npz`
  - `sequence/frame_*.obj`
  - `final/run_summary.json`
- 数据流：把外部 SMPLer-X 的逐帧输出转为 stage2 可消费的统一资产。

### `src/avatar_gs/cli/stage1_align_check.py`
- 职责：对评估脚本的 CLI 包装。
- 接口：`main()`。
- 输入：命令行参数。
- 输出/下游：转发到 stage1 silhouette 评估。

### `src/avatar_gs/cli/stage2_train_gaussian.py`
- 职责：stage2 主入口，负责读取配置、选择帧、加载 stage1 资产、构建监督和 mesh runtime，然后进入训练循环。
- 接口：`main()`。
- 主要内部接口：
  - 路径解析：`_resolve_project_path`
  - stage1 过滤：`_filter_frames_by_stage1_align`、`_load_stage1_align_scores`
  - checkpoint 对齐：`_align_checkpoint_camera_tensor`、`_load_stage2_init_checkpoint`
- 输入：`configs/stage2_gaussian.yaml`、stage1 输出、可选 stage2 init ckpt。
- 输出/下游：`outputs/stage2/gaussians3d.pt`、`gaussians3d_inner.pt`、监督缓存/摘要、outer 质量日志。
- 数据流：`config -> frame select -> rigid align -> init gaussians -> supervision state -> mesh runtime -> train_loop`。

### `src/avatar_gs/cli/render.py`
- 职责：从 stage2 checkpoint 进行真实姿态、前景-only、deform 三种渲染，并支持多种调试覆盖。
- 接口：`main()`。
- 关键内部接口：
  - 帧/姿态驱动：`_build_pose_drive_indices`、`_resolve_pick_index`
  - 读取 stage2 对齐：`_load_stage2_smpl_align_subset`
  - 图像与输出：`read_image_rgb`、`save_rgb`、`_make_layer_output_dirs`
  - 分层选择：`_split_gaussians_by_shell_layer`、`_select_gaussians`、`_clone_gaussians`
  - 可视化：`_overlay_keypoints`、`_overlay_smpl_skeleton`、`_render_points_debug`
  - deform 配置：`_resolve_deform_mesh_cfg`、`_find_deform_mesh_fallback`
- 输入：stage2 checkpoint、stage1 mesh/params、原图、keypoints、可选 skeleton NPZ。
- 输出/下游：`outputs/renders/real|fg_only|deform/{combined,inner,outer}/*.png`。
- 数据流：读取高斯与 mesh -> 可选 deform/repose -> 分层渲染 -> 可选 overlay -> 写图。

### `src/avatar_gs/cli/edit.py`
- 职责：对 stage2 checkpoint 做轻量编辑后重新渲染。
- 接口：`main()`；辅助函数 `_apply_simple_edit`。
- 输入：checkpoint、编辑配置（平移、缩放、颜色、透明度）。
- 输出/下游：编辑后的渲染结果；适合快速可视化，不是完整编辑系统。

## alignment 模块

### `src/avatar_gs/alignment/silhouette.py`
- 职责：mask/silhouette 几何工具箱。
- 接口：`mask_bbox`、`mask_outside_ratio`、`mask_iou`、`apply_uv_similarity`、`apply_similarity_to_camera_points`、`estimate_mask_similarity_transform`。
- 输入：二值 mask、顶点投影 UV、faces、图像尺寸。
- 输出/下游：对齐分数、相似变换参数、投影 mask；供评估和 stage2 对齐使用。

### `src/avatar_gs/alignment/mask_sources.py`
- 职责：统一 stage1/stage2/eval 的人物 mask 来源解析。
- 接口：
  - 投影/栅格化：`project_vertices_to_image`、`rasterize_uv_mask`、`rasterize_mesh_mask`
  - 相机：`read_stage1_frame_camera_K`
  - mask 清洗/判断：`clean_person_mask_eval_style_np`、`is_person_mask_plausible`
  - mask 来源：`load_person_mask_from_npz`、`estimate_person_mask_chromakey`、`estimate_person_mask_modnet_eval_style_np`、`resolve_person_mask_source_np`
- 输入：stage1 相机、顶点、NPZ mask、图像、MODNet。
- 输出/下游：统一的人物 mask，供 stage2 监督和评估使用。

## data 模块

### `src/avatar_gs/data/video_reader.py`
- 职责：OpenCV 视频迭代器。
- 接口：`VideoReader` 类。
- 输入：视频路径。
- 输出/下游：逐帧输出 `(idx, frame_bgr)`，供 preprocess 使用。

### `src/avatar_gs/data/frame_dataset.py`
- 职责：把抽帧目录和 `metadata.json` 组织成 PyTorch Dataset。
- 接口：`FrameDataset` 类、`_read_rgb`。
- 输入：帧目录、元数据。
- 输出/下游：训练或预处理阶段可迭代的 RGB tensor 数据。

### `src/avatar_gs/data/keypoints_mediapipe.py`
- 职责：用 MediaPipe 提取 body + hand keypoints，并做 bbox 推断与平滑。
- 接口：主入口 `extract_keypoints_from_frames`；辅助包括 `_bbox_from_landmarks`、`_bbox_from_segmentation`、`_smooth_keypoints_dict`、`_assign_hands_to_lr`。
- 输入：帧图像序列。
- 输出/下游：关键点序列与 bbox 信息，可被 stage1 写入 `keypoints.json`。

### `src/avatar_gs/data/sequence_mesh.py`
- 职责：带缓存的 stage1 OBJ 序列 mesh 读取器。
- 接口：`SequenceMeshProvider` 类、`_parse_frame_idx_from_stem`。
- 输入：`outputs/stage1/sequence/frame_*.obj`。
- 输出/下游：按帧索引稳定提供 mesh，供 stage2/render 绑定使用。

### `src/avatar_gs/data/camera_utils.py`
- 职责：空占位文件。
- 接口：无。
- 输入：无。
- 输出/下游：没有实际逻辑。

### `src/avatar_gs/data/mask_utils.py`
- 职责：空占位文件。
- 接口：无。
- 输入：无。
- 输出/下游：没有实际逻辑。

## deform 模块

### `src/avatar_gs/deform/smplx_param_source.py`
- 职责：从 `smpl_params_sequence.npz` 或顶点缓存重建 deform mesh 顶点。
- 接口：主入口 `load_deform_smplx_vertices_from_npz`；索引和兼容辅助包括 `_resolve_npz_row_index`、`_read_param_row`、`_fit_param_dim`、`_match_direct_space_to_reference`。
- 输入：stage1 参数 NPZ、目标 frame index/name、可选参考顶点。
- 输出/下游：deform mesh 顶点，供 render 使用。

## io 模块

### `src/avatar_gs/io/mesh_export.py`
- 职责：OBJ 文本导出。
- 接口：`build_obj_face_text`、`export_obj`。
- 输入：顶点数组、面数组。
- 输出/下游：OBJ 文件；主要给 stage1 导出 `sequence/frame_*.obj`。

## losses 模块

### `src/avatar_gs/losses/robust.py`
- 职责：robust photometric loss。
- 接口：`huber_loss`、`charbonnier_loss`、`compute_robust_photo_loss`。
- 输入：预测与 GT 差值。
- 输出/下游：训练循环中的光度损失。

### `src/avatar_gs/losses/regularizers.py`
- 职责：透明度、协方差、offset、颜色带、层平衡等正则项。
- 接口：`opacity_prior_loss`、`covariance_penalty`、`canonical_offset_l2`、`outer_color_band_loss`、`layer_alpha_balance_loss`。
- 输入：高斯参数与中间量。
- 输出/下游：stage2 训练 regularizer。

## segmentation 模块

### `src/avatar_gs/segmentation/modnet.py`
- 职责：MODNet runtime 封装，处理权重定位、状态字典兼容、单例缓存与人物 mask 推理。
- 接口：`resolve_modnet_ckpt_path`、`estimate_person_mask_modnet`；内部类 `_MODNetRuntime`。
- 输入：图像、可选 ckpt 路径、设备。
- 输出/下游：清洗后的人像 mask；供 stage1/stage2/eval 复用。

## smpl 模块

### `src/avatar_gs/smpl/lbs.py`
- 职责：占位性的 dummy LBS。
- 接口：`linear_blend_skinning`。
- 输入：顶点、pose。
- 输出/下游：当前实现不应作为生产逻辑使用。

### `src/avatar_gs/smpl/smpl_model.py`
- 职责：占位性的 DummySMPL 模型。
- 接口：`DummySMPL` 类。
- 输入：假 SMPL 参数。
- 输出/下游：随机/占位顶点，不是正式实现。

## gaussian 基础层

### `src/avatar_gs/gaussian/structured_gaussian.py`
- 职责：定义 `Gaussian3D` 数据结构与 checkpoint 序列化形态。
- 接口：`Gaussian3D` 类。
- 输入：位置、尺度、旋转、颜色、透明度，以及 binding/shell/quality gate 相关 tensor。
- 输出/下游：整个 stage2 和 render 共享的高斯容器。

### `src/avatar_gs/gaussian/init_from_smpl.py`
- 职责：从 mesh 初始化高斯。
- 接口：`load_obj_mesh`、`load_obj_vertices`、`init_gaussians_from_mesh_3d`。
- 输入：初始 OBJ mesh、采样数量、outer/inner 预算配置。
- 输出/下游：初始 `Gaussian3D`，供 stage2 训练起点。

### `src/avatar_gs/gaussian/losses.py`
- 职责：简单局部 loss 辅助。
- 接口：`l2_loss`。
- 输入：tensor。
- 输出/下游：少量训练逻辑复用。

### `src/avatar_gs/gaussian/rigid_align.py`
- 职责：刚体对齐的几何工具层。
- 接口：`euler_deg_to_rotmat_np`、`apply_rigid_transform_np`、`coordinate_descent_rigid_search`。
- 输入：顶点、旋转平移搜索空间、mask 评价回调所需数据。
- 输出/下游：最优刚体修正参数，供 stage2 对齐与 render 继承。

### `src/avatar_gs/gaussian/shell_policy.py`
- 职责：outer spawn/prune 与 inner softfreeze/hardprune 规则。
- 接口：`estimate_inner_count`、`estimate_outer_budget`、`compute_shell_score`、`select_outer_spawn_candidates`、`select_inner_softfreeze_candidates`、`select_inner_hardprune_candidates`、`select_outer_prune_candidates`。
- 输入：可见性、支持度、残差、score、预算参数。
- 输出/下游：训练循环中的壳层增删决策。

### `src/avatar_gs/gaussian/train_loop.py`
- 职责：stage2 真正的核心执行器。
- 接口：公开入口 `train_gaussians_3d_learnable_camera()`；实际实现 `_train_gaussians_3d_learnable_camera_impl()`；另有 checkpoint、质量门、mask 操作等大量辅助函数。
- 输入：
  - `Gaussian3D`
  - 相机张量
  - `Stage2TrainingSupervisionState`
  - `Stage2MeshRuntime` / binding state
  - stage2 配置对象
- 输出/下游：
  - 更新后的高斯参数
  - `gaussians3d.pt`
  - `gaussians3d_inner.pt`
  - outer 质量日志与报告
- 数据流：每 iter 完成 binding 更新 -> 渲染 -> loss -> optimizer -> densify/shell policy -> logging/checkpoint。

## gaussian binding 层

### `src/avatar_gs/gaussian/binding.py`
- 职责：binding_layers 的兼容导出层。
- 接口：重新暴露 binding API。
- 输入：内部 binding 模块。
- 输出/下游：给旧调用路径兼容使用。

### `src/avatar_gs/gaussian/binding_layers/types.py`
- 职责：定义 binding 配置和帧数据结构。
- 接口：`BindingModeConfig`、`BindingFrameData`。
- 输入：解析后的 binding 配置、mesh 几何张量。
- 输出/下游：作为 mesh runtime 与 per-frame binding 更新之间的数据契约。

### `src/avatar_gs/gaussian/binding_layers/frame_builder.py`
- 职责：从 mesh 构建邻接、法线、face frame、曲率方向等 binding 几何上下文。
- 接口：`build_binding_frame_data`、`parse_binding_mode_config`、`export_binding_mode_config`、`choose_default_binding_mode`，以及多种几何计算辅助函数。
- 输入：顶点、faces、binding 配置。
- 输出/下游：`BindingFrameData`，供 inner/outer binding 更新使用。

### `src/avatar_gs/gaussian/binding_layers/binding_update.py`
- 职责：binding 共用数学层。
- 接口：`quat_from_rotmat_torch`、`slerp_quat_torch`、`temporal_stabilize_vertex_dirs`、`build_vertex_binding_targets`。
- 输入：旋转矩阵、四元数、barycentric、顶点几何。
- 输出/下游：inner/outer binding 更新所需的目标姿态与方向。

### `src/avatar_gs/gaussian/binding_layers/inner_binding.py`
- 职责：inner 层按帧绑定更新。
- 接口：`compute_inner_bind_mask`、`apply_inner_binding_per_frame`、`update_inner_gaussians_with_binding`。
- 输入：`Gaussian3D`、`BindingFrameData`、绑定状态。
- 输出/下游：更新后的 inner 高斯位置与朝向。

### `src/avatar_gs/gaussian/binding_layers/outer_binding.py`
- 职责：outer 层按帧绑定更新。
- 接口：`compute_outer_anchor_mask`、`apply_outer_anchor_per_frame`、`update_outer_gaussians_with_binding`。
- 输入：`Gaussian3D`、`BindingFrameData`、face anchor/offset 信息。
- 输出/下游：更新后的 outer 高斯位置与局部偏移。

## gaussian densify 与 visibility

### `src/avatar_gs/gaussian/densify/__init__.py`
- 职责：densify 规则导出层。
- 接口：包导出。
- 输入：内部模块。
- 输出/下游：供训练主循环导入。

### `src/avatar_gs/gaussian/densify/stats.py`
- 职责：记录每个高斯的 densify 统计量。
- 接口：`GaussianDensifyStats` 类。
- 输入：可见次数、残差 EMA。
- 输出/下游：供 densify candidate 选择。

### `src/avatar_gs/gaussian/densify/rules.py`
- 职责：根据统计量更新 densify 状态并选择候选。
- 接口：`update_densify_stats`、`select_densify_candidates`。
- 输入：残差、可见性、top-percent 参数。
- 输出/下游：训练中 densify 候选索引。

### `src/avatar_gs/gaussian/visibility/__init__.py`
- 职责：visibility 导出层。
- 接口：包导出。
- 输入：内部模块。
- 输出/下游：供训练主循环调用。

### `src/avatar_gs/gaussian/visibility/gaussian_visibility.py`
- 职责：估计每个高斯在当前视角下的可见性。
- 接口：`compute_gaussian_visibility`；辅助 `_project_points`。
- 输入：高斯中心、相机 K/R/t、可选 alpha/person mask。
- 输出/下游：可见性分数、供 shell policy 和 pruning 使用。

### `src/avatar_gs/gaussian/visibility/border_attenuation.py`
- 职责：构造边界衰减图并在投影点上采样边界权重。
- 接口：`build_border_attenuation`、`sample_border_weight`。
- 输入：二值 mask、边界宽度、投影 uv。
- 输出/下游：边界衰减权重，供 photo loss 和 regularizer 使用。

## renderer 模块

### `src/avatar_gs/renderer/gaussian_renderer.py`
- 职责：对 `gsplat` 的实际渲染封装。
- 接口：`render_gaussians_3d`；内部辅助 `_attach_cached_gsplat_cuda_extension`、`_ensure_R`、`_ensure_t`、`_ensure_K`、`_make_viewmats`。
- 输入：`Gaussian3D`、相机内外参、背景图/颜色、图像大小。
- 输出/下游：RGB、alpha、meta；供训练和 render 共用。

### `src/avatar_gs/renderer/camera.py`
- 职责：空占位文件。
- 接口：无。
- 输入：无。
- 输出/下游：没有实际逻辑。

## stage2 配置/运行时/监督/报告层

### `src/avatar_gs/stage2/config.py`
- 职责：把原始 YAML 拆成强结构配置对象。
- 接口：`Stage2FrameSelectionConfig`、`Stage2BindingConfig`、`Stage2SupervisionConfig`、`Stage2TrainConfig`、`Stage2TrainLoopConfig`。
- 输入：原始 stage2 yaml 字典。
- 输出/下游：供 `stage2_train_gaussian.py`、`train_loop.py`、`mesh_runtime.py` 使用的结构化配置。

### `src/avatar_gs/stage2/mesh_runtime.py`
- 职责：统一 stage2 训练和 render 时的 mesh 运行时上下文。
- 接口：
  - 路径与索引：`resolve_effective_rigid_align_config`、`resolve_mesh_sequence_dir`、`resolve_bind_mesh_indices`
  - 邻接/法线：`compute_vertex_normals_np`、`pack_face_neighbor_arrays`、`compute_face_neighbor_weights`
  - 运行时类：`Stage2MeshContext`、`Stage2MeshRuntime`、`Stage2TrainingBindingState`
  - 初始化：`initialize_stage2_training_binding_state`
- 输入：stage1 OBJ/NPZ、frames、rigid align 配置、binding 配置。
- 输出/下游：当前帧 mesh context、binding frame data、训练绑定状态；供 train_loop 和 render 使用。

### `src/avatar_gs/stage2/rigid_align.py`
- 职责：stage2 级别的 rigid align 配置标准化与搜索驱动。
- 接口：`normalize_rigid_align_config`、`run_stage2_rigid_align_search`、`Stage2RigidAlignResult`。
- 输入：stage1 顶点/mesh、人物 mask、采样帧、搜索参数。
- 输出/下游：刚体修正配置和评估指标，供 stage2 训练与 render 继承。

### `src/avatar_gs/stage2/supervision.py`
- 职责：stage2 监督系统本体。
- 接口：
  - 投影/栅格化：`project_points_numpy`、`project_points_torch`、`rasterize_mesh_mask`
  - mask/contour：`estimate_gt_person_contour_mask_np`、`modnet_person_mask_eval_style_np`、`compute_contour_band_torch`、`fuse_outer_contour_mask_np`
  - 资产层：`Stage2FrameAssets`、`build_stage2_frame_assets`
  - 缓存：`load_stage2_supervision_cache`、`prepare_stage2_supervision_artifacts`
  - 训练态：`Stage2TrainingSupervisionState`、`build_stage2_training_supervision_state`
- 输入：原图、stage1 相机与 mesh、人物 mask、MODNet、配置。
- 输出/下游：inner/outer/combined RGB target、SMPL mask、GT contour、frame weights、缓存与预览图；供 train_loop 使用。

### `src/avatar_gs/stage2/reporting.py`
- 职责：stage2 outer 质量评估与报告输出。
- 接口：`default_stage2_quality_thresholds`、`compute_outer_dynamic_caps`、`prepare_stage2_report_paths`、`evaluate_stage2_quality`、`write_stage2_quality_log`、`write_stage2_json_report`、`build_outer_face_records`、`build_outer_spawn_report`。
- 输入：训练过程中的可见性、support、outside ratio 等指标。
- 输出/下游：`outer_quality_metrics.jsonl`、`outer_quality_report.json`、spawn/face 报表。

## training helper 层

### `src/avatar_gs/training/frame_weighting.py`
- 职责：按帧残差维护 EMA 并计算 frame weight。
- 接口：`FrameResidualEMA`、`frame_weight_from_residual`。
- 输入：每帧残差。
- 输出/下游：帧权重，供监督和训练循环使用。

### `src/avatar_gs/training/inner_layer_training.py`
- 职责：解析 inner 实际 supervision mask。
- 接口：`resolve_inner_supervision_mask`。
- 输入：SMPL mask、人物 mask、配置。
- 输出/下游：inner 训练时真正使用的 mask。

### `src/avatar_gs/training/outer_layer_training.py`
- 职责：outer trigger ring 与 face gap target 的生成逻辑。
- 接口：`build_outer_trigger_mask`、`map_trigger_ring_to_faces`、`estimate_face_gap_targets`。
- 输入：GT contour/person mask、SMPL mask、mesh 法线/face 邻域、配置。
- 输出/下游：outer trigger ring、face 激活、gap target；供 train_loop 中 outer 训练阶段使用。

## utils 层

### `src/avatar_gs/utils/cfg.py`
- 职责：YAML 读取、深更新、命令行 override 解析与保存。
- 接口：`load_yaml`、`deep_update`、`resolve_path`、`parse_args_base`、`apply_overrides`、`save_yaml`。
- 输入：yaml 文件与 override 字符串。
- 输出/下游：统一配置字典；供所有 CLI 使用。

### `src/avatar_gs/utils/io.py`
- 职责：目录创建与 JSON 读写。
- 接口：`ensure_dir`、`read_json`、`write_json`。
- 输入：路径、对象。
- 输出/下游：被 stage1/stage2/render 反复复用。

### `src/avatar_gs/utils/log.py`
- 职责：构造统一 logger。
- 接口：`setup_logger`。
- 输入：logger 名称、文件路径、级别。
- 输出/下游：CLI 与训练过程日志。

### `src/avatar_gs/utils/person_contour.py`
- 职责：兼容 shim，把 contour 逻辑转发到 `evaluation/stage1_eval/person_contour.py`。
- 接口：模块级转发。
- 输入：调用方 import。
- 输出/下游：避免多份 contour 实现分叉。

### `src/avatar_gs/utils/seed.py`
- 职责：固定随机种子，并设置 deterministic 相关选项。
- 接口：`seed_everything`。
- 输入：seed。
- 输出/下游：保证训练和实验复现性。

## `src/lib` 轻量 vendored 模块

### `src/lib/__init__.py`
- 职责：标记 vendored motion-prior 支撑模块。
- 接口：包占位。
- 输入：无。
- 输出/下游：包导入路径。

### `src/lib/THIRD_PARTY_LICENSE_STAGE1_MOTION_PRIOR.txt`
- 职责：记录该 vendored 模块的许可证。
- 接口：文档。
- 输入：无。
- 输出/下游：合规说明。

### `src/lib/core/__init__.py`
- 职责：空导出。
- 接口：无。
- 输入：无。
- 输出/下游：包结构占位。

### `src/lib/core/config.py`
- 职责：解析 motion prior 数据目录环境变量。
- 接口：`_default_motion_prior_data_dir`。
- 输入：环境变量。
- 输出/下游：给 motion prior 相关流程提供默认目录。

### `src/lib/data_utils/__init__.py`
- 职责：空导出。
- 接口：无。
- 输入：无。
- 输出/下游：包占位。

### `src/lib/data_utils/img_utils.py`
- 职责：旧人姿管线的图像裁剪、bbox 变换、归一化、序列切块工具。
- 接口：`get_image`、`generate_patch_image_cv`、`crop_image`、`transfrom_keypoints`、`get_single_image_crop_demo`、`split_into_chunks` 等。
- 输入：图像、bbox、关键点。
- 输出/下游：motion prior / SPIN 风格模型的图像 patch 与几何变换结果。

### `src/lib/dataset/__init__.py`
- 职责：空导出。
- 接口：无。
- 输入：无。
- 输出/下游：包占位。

### `src/lib/dataset/inference.py`
- 职责：为 motion prior / SPIN 构造推理数据集。
- 接口：`Inference`、`ImageFolder`。
- 输入：图片目录、bbox 或 2D joints。
- 输出/下游：可迭代数据集，供旧式模型推理使用。

### `src/lib/models/__init__.py`
- 职责：空导出。
- 接口：无。
- 输入：无。
- 输出/下游：包占位。

### `src/lib/models/motion_prior.py`
- 职责：时序 GRU 编码器与 motion prior 网络。
- 接口：`TemporalEncoder`、`MotionPriorNet`、`MotionPriorDemo`。
- 输入：时序姿态/特征。
- 输出/下游：平滑后的姿态先验特征或参数。

### `src/lib/models/smpl.py`
- 职责：基于 `smplx.SMPL` 扩展更多关节点的 SMPL 模型包装。
- 接口：`SMPL`、`get_smpl_faces`。
- 输入：SMPL 参数。
- 输出/下游：顶点、关节、faces；供旧兼容链路使用。

### `src/lib/models/spin.py`
- 职责：SPIN/HMR 主体网络与回归器。
- 接口：`Bottleneck`、`HMR`、`Regressor`、`hmr`、`projection`、`perspective_projection`、`get_pretrained_hmr`。
- 输入：人体裁剪图像。
- 输出/下游：SMPL 参数、相机参数、2D/3D 投影结果。

### `src/lib/utils/__init__.py`
- 职责：空导出。
- 接口：无。
- 输入：无。
- 输出/下游：包占位。

### `src/lib/utils/geometry.py`
- 职责：Rodrigues、四元数、旋转矩阵、平移估计、6D rotation 转换等底层几何函数。
- 接口：`batch_rodrigues`、`quat2mat`、`rotation_matrix_to_angle_axis`、`estimate_translation`、`rot6d_to_rotmat` 等。
- 输入：旋转/位姿张量、3D/2D 点。
- 输出/下游：旧式人体姿态链路的几何运算。

### `src/lib/utils/one_euro_filter.py`
- 职责：One Euro Filter 实现。
- 接口：`OneEuroFilter` 类及其基础函数。
- 输入：时序信号。
- 输出/下游：平滑后的信号。

### `src/lib/utils/smooth_bbox.py`
- 职责：从关键点估计并平滑 bbox 参数。
- 接口：`get_smooth_bbox_params`、`kp_to_bbox_param`、`get_all_bbox_params`、`smooth_bbox_params`。
- 输入：关键点序列。
- 输出/下游：更稳定的 bbox，供裁剪与推理使用。

### `src/lib/utils/smooth_pose.py`
- 职责：对时序 pose 用 One Euro Filter 做平滑，再经 SMPL 重建平滑后的顶点/关节。
- 接口：`smooth_pose`。
- 输入：pose、betas。
- 输出/下游：平滑后的 pose/verts/joints。

## 数据与输出契约

### 输入数据
- `data/videos/input.mp4`
  - 职责：默认输入视频。
  - 下游：`preprocess_video.py`。
- `data/frames/input/metadata.json`
  - 职责：抽帧元数据。
  - 下游：stage1、`FrameDataset`。
- `data/frames/input/*.jpg`
  - 职责：抽帧图像。
  - 下游：stage1、render 背景读取、评估。

### Stage1 输出
- `outputs/stage1/cameras.json`
  - 内容：每帧相机内参与图像信息。
  - 下游：stage2 supervision、render、eval。
- `outputs/stage1/keypoints.json`
  - 内容：每帧关键点。
  - 下游：render overlay、自动评估辅助。
- `outputs/stage1/person_masks.npz`
  - 内容：人物 mask 序列。
  - 下游：stage2 supervision、rigid align、eval。
- `outputs/stage1/smpl_params_sequence.npz`
  - 内容：逐帧 SMPL-X 参数。
  - 下游：render deform、部分绑定/兼容逻辑。
- `outputs/stage1/smpl_vertices_sequence.npz`
  - 内容：逐帧顶点序列。
  - 下游：stage2 mesh 绑定、eval。
- `outputs/stage1/smpl_joints_sequence.npz`
  - 内容：逐帧关节序列。
  - 下游：render skeleton。
- `outputs/stage1/smpl_skeleton_sequence.npz`
  - 内容：骨架映射序列。
  - 下游：render overlay。
- `outputs/stage1/sequence/frame_*.obj`
  - 内容：逐帧 mesh OBJ。
  - 下游：stage2 mesh runtime、render。
- `outputs/stage1/final/run_summary.json`
  - 内容：stage1 汇总。
  - 下游：人工查看与排障。

### Stage2 输出
- `outputs/stage2/gaussians3d.pt`
  - 内容：主 checkpoint。
  - 下游：render、PLY 导出、后续训练 warm start。
- `outputs/stage2/gaussians3d_inner.pt`
  - 内容：inner 阶段 checkpoint。
  - 下游：分阶段训练或调试。
- `outputs/stage2/supervision_cache.npz`
  - 内容：离线监督缓存。
  - 下游：训练直接加载。
- `outputs/stage2/supervision_summary.json`
  - 内容：监督构建摘要。
  - 下游：人工排障。
- `outputs/stage2/outer_quality_metrics.jsonl`
  - 内容：outer 训练过程质量日志。
  - 下游：质量分析。
- `outputs/stage2/outer_quality_report.json`
  - 内容：最终 outer 质量报告。
  - 下游：验收判断。
- `outputs/stage2/outer_spawn_per_frame.json`
  - 内容：trigger/spawn 记录。
  - 下游：outer 行为分析。
- `outputs/stage2/supervision_preview/**`
  - 内容：监督预览图。
  - 下游：人工核查 supervision 是否跑偏。

### Render 输出
- `outputs/renders/real/{combined,inner,outer}/*.png`
  - 内容：真实姿态渲染。
  - 下游：视觉验收。
- `outputs/renders/fg_only/{combined,inner,outer}/*.png`
  - 内容：去背景渲染。
  - 下游：评估和前景分析。
- `outputs/renders/deform/{combined,inner,outer}/*.png`
  - 内容：deform 或标准姿态渲染。
  - 下游：形变和编辑检查。

### Stage1 评估输出
- `evaluation/output_images/stage1/*_gt_stage1_contour.png`
  - 内容：GT 与 stage1 contour 叠图。
  - 下游：人工查看对齐质量。
- `evaluation/output_images/stage1/*_stage1_eval_report.json`
  - 内容：单帧评估报告。
  - 下游：对齐筛选。
- `evaluation/output_images/stage1/stage1_eval_report_all.json`
  - 内容：多帧聚合报告。
  - 下游：stage2 帧过滤和质量判断。

## 第三方目录边界

### `third_party/SMPLer-X/**`
- 职责：stage1 的主要人体参数估计后端。
- 接口：仓库级外部依赖；本仓库主要通过 `third_party/SMPLer-X/main/inference.py` 调用。
- 输入：选定帧图像、模型权重。
- 输出/下游：SMPL-X JSON/NPZ/可视化结果，供 `stage1_fit_smplerx.py` 整理。

### `third_party/MODNet/**`
- 职责：人像 matting 外部依赖。
- 接口：本仓库通过 `src/avatar_gs/segmentation/modnet.py` 间接调用其模型定义和权重。
- 输入：RGB 图像。
- 输出/下游：人物 alpha/mask，供 stage1/stage2/eval 统一使用。

## 总数据流

1. `preprocess_video.py`
   `input.mp4 -> frame jpg + metadata.json`
2. `stage1_fit_smplerx.py`
   `frame jpg + SMPLer-X -> cameras/keypoints/masks/params/verts/obj`
3. `stage2_train_gaussian.py`
   `stage1 assets + stage2 config -> Gaussian3D + supervision + mesh runtime -> gaussians3d.pt`
4. `render.py`
   `gaussians3d.pt + stage1 mesh/params/images -> real/fg_only/deform png`
5. `run_stage1_eval.py`
   `stage1 verts/cameras + GT/person mask -> contour metrics/report`

## 负责人维护优先级

高优先级文件：
- `src/avatar_gs/cli/stage1_fit_smplerx.py`
- `src/avatar_gs/cli/stage2_train_gaussian.py`
- `src/avatar_gs/cli/render.py`
- `src/avatar_gs/gaussian/train_loop.py`
- `src/avatar_gs/stage2/mesh_runtime.py`
- `src/avatar_gs/stage2/supervision.py`
- `src/avatar_gs/training/outer_layer_training.py`
- `configs/stage2_gaussian.yaml`

原因很简单：这些文件共同决定这个仓库到底在做什么，而不是你希望它在做什么。
