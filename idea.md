# idea.md

## 仓库当前真相

- 这个仓库不是“一个干净统一的从视频到可编辑 Avatar 的产品化实现”，而是一个还在演化中的工程拼装体。
- 真正需要负责人理解和维护的是手写源码、配置、脚本、评估入口，大约 90 个文件级别；`third_party/`、`outputs/`、`data/` 里大量文件是外部依赖、缓存、产物、可视化，不适合伪装成“逐文件同权维护对象”。
- 如果你把第三方整仓和逐帧产物也按同等粒度讲解，那不是负责，是失去优先级。

## 主执行链

1. `src/avatar_gs/cli/preprocess_video.py`
   从 `data/videos/input.mp4` 抽帧，写入 `data/frames/input/*.jpg` 和 `metadata.json`。
2. `src/avatar_gs/cli/stage1_fit_smplerx.py`
   调 SMPLer-X 做逐帧人体参数估计，并整理成 stage2 可消费资产。
3. `src/avatar_gs/cli/stage2_train_gaussian.py`
   读取 stage1 资产，初始化高斯，建立监督与绑定，进入训练循环。
4. `src/avatar_gs/cli/render.py`
   从 stage2 checkpoint 生成 `real / fg_only / deform` 渲染结果。
5. 可选：`evaluation/stage1_eval/run_stage1_eval.py`
   对 stage1 的 silhouette 对齐质量做单帧或多帧评估。

## 必须写明白的分叉点

### 分叉 1：Stage1 的真实身份和你以为的身份不一样

- 名字像“SMPL 拟合阶段”，真实角色更接近“SMPLer-X 结果整理与导出阶段”。
- 代码并没有把主要精力放在自研优化器上，而是：
  - 准备 SMPLer-X 权重和运行目录。
  - 调 `cli/_run_smplerx_refine_frames.py` 跑外部推理。
  - 从 SMPLer-X 的 JSON/NPZ 重建 `verts_cam`、`joints_cam`、相机内参。
  - 估计/融合人物 mask。
  - 输出 stage2 所需的 `cameras.json`、`keypoints.json`、`person_masks.npz`、`smpl_params_sequence.npz`、`smpl_vertices_sequence.npz`、`sequence/frame_*.obj`。
- 结论：stage1 的成败高度依赖 `third_party/SMPLer-X` 和外部 checkpoint，不是本仓库独立完成的人体拟合。

### 分叉 2：人物 mask 有多路来源，不是一条真理链

人物监督不是单一来源，而是多路分支后再汇合：

- stage1 路线：
  - `chromakey` 绿色背景抠图。
  - `MODNet` 人像 matting。
  - SMPL 投影得到的 mesh hint mask 作为参考。
- stage2 路线：
  - 直接读 `outputs/stage1/person_masks.npz`。
  - 按配置重跑 MODNet。
  - `person_mask_fuse_mode` 决定是 `intersect`、`union` 还是偏向某一路。
- eval 路线：
  - `evaluation/stage1_eval/person_contour.py` 重新读 mask / 跑 MODNet / 提 contour。

这意味着一个残酷事实：很多你以为的“几何错误”，其实是 mask 来源分叉后监督质量不一致导致的。

### 分叉 3：Stage2 绑定源也有分叉

高斯不是静态漂在空间里，它们要绑定到每一帧 mesh。但 mesh 来源也有分叉：

- `outputs/stage1/sequence/frame_*.obj`
- `outputs/stage1/smpl_vertices_sequence.npz`
- render 阶段还可以走 deform mesh 重建路径

`stage2/mesh_runtime.py` 负责把这些来源统一成运行时 mesh 上下文，再喂给 binding 更新逻辑。

### 分叉 4：内层和外层不是同一种东西

- inner gaussians：贴体、稳定、保守，优先还原主体本体。
- outer gaussians：补偿头发、衣物边缘、离体残差。
- inner 与 outer 绑定策略不同：
  - inner 偏 vertex 主导，必要时 face fallback。
  - outer 偏 face 主导，并带 `bind_face_offset_local` 的局部偏移。
- 这不是“一个高斯模型调几个参数”，而是已经分成两套行为系统。

### 分叉 5：outer 监督机制已经换代

- 旧的 residual-only 叙事已经过时。
- 当前 outer 监督入口是 `training/outer_layer_training.py::build_outer_trigger_mask`。
- 逻辑是：
  - 从 GT/person contour 得到 `outer_base_mask`
  - 用 `dilate(SMPL)` 排除贴体区域
  - 形成 trigger ring
  - 视配置叠加 edge gate
  - 像素太少时 relaxed fallback
  - 可附加 detached solid 区域
  - 再映射到 mesh face，得到外层 face 级激活与 gap target

也就是说，当前 outer 是“由 contour-trigger 驱动的 face 级局部壳层训练”，不是老版本那种泛化的 residual 噪声池。

### 分叉 6：代码能力和默认运行模式已经分叉

这是现在最重要、也最容易被忽视的一点。

- `configs/stage2_gaussian.yaml` 当前默认写的是 `training_mode: inner_only`。
- 但仓库的大量代码、术语、参数设计都在强调 dual-layer / outer shell。
- 结果是：
  - 代码结构在讲“完整双层系统”。
  - 默认命令却只跑 inner phase。
  - `scripts/auto_full_pipeline.ps1` 名字像 full pipeline，实际上默认也会调这份 `inner_only` 主配置。

这不是小瑕疵，这是认知分叉。你如果不正面承认，就会一直误判自己到底验证了什么。

### 分叉 7：监督可以在线算，也可以离线缓存

`stage2/supervision.py` 现在存在两条路径：

- 在线构建监督：训练前或训练时现算各种 mask / contour / RGB target。
- 离线缓存监督：先跑 `--prepare_supervision_only`，生成 `outputs/stage2/supervision_cache.npz`，之后训练直接加载。

这意味着监督逻辑和训练逻辑已经部分解耦。好处是可复现，坏处是缓存一旦过期或和配置不一致，就会制造“看起来训练稳定，实际在吃旧监督”的假象。

### 分叉 8：rigid align 已经前置化和运行时化

- `stage2/rigid_align.py` 负责搜索刚体修正配置。
- `gaussian/rigid_align.py` 提供几何计算。
- `stage2/mesh_runtime.py` 和 render 阶段继续应用该刚体修正。

这直接说明作者已经接受一个事实：stage1 导出的 mesh 与真实图像并不天然对齐，必须承认并补偿。

### 分叉 9：render 已经不是纯展示层，而是第二执行器

`cli/render.py` 现在承担的职责包括：

- 选择 frame / pose-drive 索引
- 读取 deform SMPL-X 参数或 mesh
- 构造标准姿态或指定帧 mesh
- 渲染 `real / fg_only / deform`
- six-view 输出
- inner / outer / combined 分层输出
- skeleton / keypoint / debug overlay
- 可见性 override 与 bound-only 筛选

说白了，render 已经长成一个第二执行器，而不是一个薄壳 viewer。

### 分叉 10：仓库里存在占位文件和历史兼容壳

以下文件不是主逻辑，而是会误导读者的占位/兼容残留：

- `src/avatar_gs/__init__.py`
- `src/avatar_gs/data/camera_utils.py`
- `src/avatar_gs/data/mask_utils.py`
- `src/avatar_gs/renderer/camera.py`
- `src/avatar_gs/smpl/lbs.py`
- `src/avatar_gs/smpl/smpl_model.py`

前四个基本空壳，后两个是 dummy 实现。它们的危险不在于会运行，而在于会制造错误预期。

## 现在这套代码真正怎么运转

### Stage1 数据流

- 输入：`data/frames/input/*.jpg`
- 经过：SMPLer-X 推理、mask 估计、参数重建、OBJ 导出
- 输出：
  - `outputs/stage1/cameras.json`
  - `outputs/stage1/keypoints.json`
  - `outputs/stage1/person_masks.npz`
  - `outputs/stage1/smpl_params_sequence.npz`
  - `outputs/stage1/smpl_vertices_sequence.npz`
  - `outputs/stage1/smpl_joints_sequence.npz`
  - `outputs/stage1/smpl_skeleton_sequence.npz`
  - `outputs/stage1/sequence/frame_*.obj`

### Stage2 数据流

- 读取 stage1 资产与主配置
- 解析帧选择、刚体对齐、mesh 绑定来源
- 从初始 mesh 采样生成 `Gaussian3D`
- 建立 `Stage2TrainingSupervisionState`
- 每轮迭代：
  - 取当前帧 mesh context
  - inner/outer 分别做 binding 更新
  - 渲染预测
  - 计算 photo / robust / regularizer / silhouette / outer penalty
  - 更新 densify 统计与 shell policy
  - 写质量日志和 checkpoint

### Render 数据流

- 输入：`outputs/stage2/gaussians3d.pt` + stage1 mesh / params / images
- 经过：frame 选择、可选 deform mesh 重建、分层高斯选择、渲染、overlay
- 输出：`outputs/renders/real|fg_only|deform/...`

## 风险排序

1. 默认配置和代码叙事分裂
   你以为自己在测完整系统，实际上常常只在测 inner-only。

2. mask 质量决定很多后续指标
   对齐、trigger ring、spawn/prune、评估都被 mask 绑架。

3. `train_loop.py` 与 `render.py` 过重
   继续堆规则，迟早会变成“任何改动都高风险”的沼泽。

4. 监督缓存与配置可能漂移
   不校验缓存来源，就可能在吃旧数据还自以为结果稳定。

5. 空壳文件和 dummy 文件制造错误认知
   新维护者会被这些名字骗进去，以为那里真的有实现。

## 负责人应该立刻做的事

1. 明确仓库默认目标。
   如果产品目标是 dual-layer avatar，就把默认训练模式修正为 `full` 或显式拆成 `stage2_inner_cache.yaml` 与 `stage2_full.yaml`。

2. 拆 `train_loop.py`。
   至少按 supervision / binding / loss / shell policy / checkpointing 拆层。继续把所有规则塞在一个文件里，是在给未来埋雷。

3. 统一 mask 真相。
   把 stage1、stage2、eval 的 mask 来源优先级写成单一文档与统一配置，不要让三套路径各说各话。

4. 清理空壳和 dummy。
   保留就明确标注 placeholder；不保留就删除。别让文件名继续撒谎。

5. 把 `auto_full_pipeline.ps1` 改名或改实。
   名字叫 full，就该真的 full；否则它只是自动化地传播误解。
