# idea.md

## Stage2 当前方案（唯一生效）

### 目标
- 内层高斯：严格贴合 SMPL-X 表面，负责主体稳定重建。
- 外层高斯：只在 GT 与 SMPL 轮廓差异区域触发，负责离体细节（头发、衣物边缘、附加物）。
- 训练优先级：先抑制背景渗入和越界噪声，再提升离体细节覆盖。

### 核心设计
1. 触发机制（外层）
- 外层监督唯一入口是 `trigger ring`。
- 基础掩码：`GT_mask & ~dilate(SMPL_mask, gap_px)`。
- 再做 ring 化（`outer_trigger_ring_width_px`）和 edge gate（`outer_trigger_edge_th`）。
- 帧级触发阈值：`trigger_pixels >= outer_trigger_min_pixels`。

2. 绑定机制（内外分离）
- 内层：`inner_binding.py`，顶点主导 + 面回退，不使用外层局部偏移。
- 外层：`outer_binding.py`，面主导 + 局部偏移 `bind_face_offset_local`（t1/t2/n）。
- 共用数学和顶点回退基础逻辑在 `binding_update.py`。

3. 训练稳定性与防失真
- `inner_use_strict_smpl_mask=true`：内层监督严格用 SMPL mask，不做边界扩张。
- 外层跨视角硬边界：越界点执行 alpha/scale/offset 硬压制，并有 patience 后硬剪枝。
- 轮廓外外层点专门惩罚：`outer_outside_opacity_weight` + `outer_outside_offset_weight`。
- `prune_state==2` 的点透明度被强压（`outer_pruned_opacity_logit_max`），避免“占位点可见噪声”。
- 可见性下限只给内层，不给外层，避免外层弱噪点被“保活”。

4. 外层增删策略（无旧重分配）
- spawn 来源只允许 inner active 且触发支持满足阈值。
- 不再使用 outer->outer/source reassign 旧链路。
- 低支持外层点按 patience 强制回收。

### 当前生效的关键参数约束
- 禁用旧键：
  - `outer_vis_floor` / `outer_vis_edge_floor`
  - `outer_edge_visibility_boost`
  - `outer_support_sample_dilate_px`
  - `outer_reassign_*`
  - `outer_spawn_allow_outer_source`
  - `outer_reassign_allow_outer_source`
  - `outer_reassign_preserve_source_offset`
  - `outer_head_min_ratio` / `outer_contour_dilate_px` / `outer_contour_band_px`
- CLI 会对上述旧键 fail-fast 报错。

### 验收标准（训练通过门槛）
- `outer_active_outside_contour_ratio <= 1.5%`
- `outer_active_outside_person_ratio <= 1.0%`
- `outer_active_outside_alpha_gt_0.05_ratio <= 0.2%`
- `hand_leakage_ratio <= 0.3%`
- `outer_arm_nontrigger_activation_ratio <= 0.5%`
- `outer_projection_outside_anyframe_ratio <= 1.0%`
- `outer_head_ring_coverage >= baseline`

### 运行命令（Windows PowerShell）
```powershell
$env:PYTHONPATH='src'
$env:TORCH_EXTENSIONS_DIR="$PWD\.torch_extensions"
python -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml
```

```powershell
$env:PYTHONPATH='src'
python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml --override render.deform_smplx_use_standard_pose=true
```

### 调参顺序（仅在新参数范围内）
1. 先控噪：`outer_outside_opacity_cap`、`outer_outside_offset_decay`、`outer_boundary_*`。
2. 再提细节：`outer_trigger_edge_th`、`outer_trigger_min_pixels`、`outer_spawn_support_th`。
3. 最后平衡纹理：`outer_silhouette_weight`、`outer_silhouette_band_weight`。

### 已知风险
- GT person mask 质量差会直接误导 trigger ring。
- 触发阈值过低会引发手臂和边缘误触发；过高会漏掉细发丝。
- 外层预算过大（`shell_max_ratio` 太高）会放大噪点重生概率。
