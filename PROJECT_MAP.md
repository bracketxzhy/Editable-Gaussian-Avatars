# PROJECT_MAP

## Scope

This map covers the files that define the repository's owned logic:
- top-level docs and dependency manifests
- `configs/**`
- `scripts/**`
- `evaluation/stage1_eval/**`
- `src/avatar_gs/**`
- `src/lib/**`

It does not attempt a line-by-line map for `third_party/**` or `outputs/**`.
That would be noise, not maintenance. For those directories, this file records boundaries and contracts instead.

For each file below, the contract is described in four parts:
- role
- interface
- inputs
- outputs or downstream users

## Current Stage1 Additions

The current stage1 implementation adds a new semantic branch, a default-on silhouette-refine postprocess, and optional shared-beta or best-of branches.
The important additions are listed here so the rest of the file can stay mostly stable.

### `third_party/FASHN/`
- Role: vendored local FASHN model snapshot and upstream metadata.
- Interface: asset directory, not a normal owned source module.
- Inputs: consumed by `src/avatar_gs/segmentation/fashn.py`.
- Outputs/downstream: parser model used by local clothing and exposed-body semantics.

### `assets/fashn_local/runtime/`
- Role: vendored importable runtime packages required by the local FASHN parser.
- Interface: runtime directory injected into `sys.path`.
- Inputs: consumed by `src/avatar_gs/segmentation/fashn.py`.
- Outputs/downstream: makes local FASHN parsing runnable without separately managed site-packages.

### `src/avatar_gs/cli/stage1_build_fashn_face_semantics.py`
- Role: standalone CLI to build face-level SMPL semantics from stage1 outputs and local FASHN parsing.
- Interface: `main()`.
- Inputs: stage1 directory, frames directory, output directory, FASHN device and parser settings.
- Outputs/downstream: writes `smpl_face_semantics.npz`, CSV, previews, and summary JSON for stage2.

### `src/avatar_gs/segmentation/fashn.py`
- Role: adapter around the vendored FASHN parser.
- Interface: model or runtime path resolution, runtime import patching, parser loading, device normalization, segmentation prediction, and coarse label mapping.
- Inputs: RGB image arrays plus local FASHN model and runtime paths.
- Outputs/downstream: raw parser labels and coarse groups consumed by stage1 semantics export.

### `src/avatar_gs/segmentation/smpl_face_semantics.py`
- Role: map frame-level FASHN predictions onto SMPL face topology.
- Interface: face-topology loading, visible-face rasterization, vote accumulation, preview export, CSV export, and summary export.
- Inputs: stage1 vertices, stage1 cameras, frame RGBs, FASHN parser predictions.
- Outputs/downstream: `smpl_face_semantics.npz` and related reports used by stage2.

### `src/avatar_gs/smpl/__init__.py`
- Role: SMPL package export layer.
- Interface: package import layer.
- Inputs: `shared_beta_refit.py` and `silhouette_refine.py`.
- Outputs/downstream: exports the shared-beta and silhouette-refine APIs to stage1.

### `src/avatar_gs/smpl/shared_beta_refit.py`
- Role: estimate one shared sequence beta and refit per-frame pose or translation while holding shape fixed.
- Interface: `SharedBetaRefitConfig`, `SharedBetaRefitResult`, `apply_shared_beta_refit`.
- Inputs: stage1 records, optional alignment IoU, optional person masks or 2D joints, body-model assets.
- Outputs/downstream: rewritten stage1 records and shared-beta metadata.

### `src/avatar_gs/smpl/silhouette_refine.py`
- Role: refine per-frame SMPLest-X parameters against person masks and silhouette quality metrics.
- Interface: `SilhouetteRefineConfig`, `SilhouetteRefineResult`, `apply_silhouette_refine`.
- Inputs: stage1 records, body-model assets, mask-source selection, and optional MODNet checkpoint.
- Outputs/downstream: rewritten stage1 records plus per-frame acceptance and loss metadata consumed by stage1 export and summaries.

### `src/avatar_gs/stage2/fashn_semantics.py`
- Role: load and validate the face-semantics NPZ for stage2.
- Interface: `Stage2FaceSemantics`, `load_stage2_face_semantics`.
- Inputs: `smpl_face_semantics.npz`, optional expected face count.
- Outputs/downstream: validated semantic asset for `src/avatar_gs/gaussian/train_loop.py`.

### `outputs/stage1/smplestx/`
- Role: raw backend runtime directory and predictions produced by the selected-frame SMPLest-X invocation.
- Interface: generated output directory.
- Inputs: written by `src/avatar_gs/cli/_run_smplestx_refine_frames.py`.
- Outputs/downstream: read by `src/avatar_gs/cli/stage1_fit_smplestx.py` when rebuilding final stage1 records.

### `outputs/stage1/fashn_face_semantics/`
- Role: default face-semantics asset directory under the stock stage1 output root.
- Interface: generated output directory.
- Inputs: written by stage1 FASHN face-semantics export.
- Outputs/downstream: provides semantic assets for stage2 semantic policies.

### `outputs/stage1_sharedbeta/`
- Role: legacy or alternate stage1 root still referenced by the stock stage2 config, not the stage1 CLI default.
- Interface: generated output directory only if the operator targets that `--out_dir` or keeps older artifacts.
- Inputs: written by `src/avatar_gs/cli/stage1_fit_smplestx.py` only when explicitly targeted there.
- Outputs/downstream: still referenced by `configs/stage2_gaussian.yaml` for geometry and mask paths.

### `outputs/stage1_bestof/`
- Role: optional merged stage1 root assembled from per-frame winners across evaluated candidate runs.
- Interface: generated output directory.
- Inputs: written by `src/avatar_gs/cli/stage1_bestof_merge.py` from candidate stage1 dirs and evaluation reports.
- Outputs/downstream: stage2-compatible alternative to a single-run `outputs/stage1`.

## Top-Level Files

### `README.md`
- Role: public-facing project overview, setup guide, reproduction guide, and usage entrypoint.
- Interface: markdown document.
- Inputs: current repository behavior and environment requirements.
- Outputs/downstream: used by new users to understand what the project claims to do.

### `requirements.txt`
- Role: Python dependency list for core runtime and stage1/stage2 integration.
- Interface: `pip install -r requirements.txt`.
- Inputs: package constraints derived from code imports.
- Outputs/downstream: base Python environment for local execution.
- Caveat: does not fully solve the environment by itself; `mmcv-full==1.7.1`, body-model assets, and editable install of `third_party/SMPLest-X/main/transformer_utils` are still required.

### `idea.md`
- Role: repository truth ledger, branch map, risk list, and maintenance priorities.
- Interface: markdown document.
- Inputs: observed code behavior.
- Outputs/downstream: helps the owner avoid believing the marketing version of the repo.

### `PROJECT_MAP.md`
- Role: file-level architecture and dataflow map.
- Interface: markdown document.
- Inputs: actual repository structure.
- Outputs/downstream: maintenance map for future code changes.

### `mmcv_build.log`
- Role: local environment build log for `mmcv` installation attempts.
- Interface: plain text log.
- Inputs: local install commands.
- Outputs/downstream: troubleshooting reference only.

### `outer`
- Role: stray top-level artifact with no meaningful repository contract.
- Interface: none.
- Inputs: none.
- Outputs/downstream: confusion.

## Environment Reality

The repo requires more than the dependency file.
Actual setup depends on:
- `requirements.txt`
- CUDA/PyTorch-compatible `torch` and `torchvision`
- `mmcv-full==1.7.1`
- editable install of `third_party/SMPLest-X/main/transformer_utils`
- external checkpoints for SMPLest-X and MODNet
- SMPL / SMPL-X body-model assets under `assets/**`

That is why the README, requirements, and stage1 docs must stay consistent.

## Config Files

### `configs/video.yaml`
- Role: preprocess configuration.
- Interface: loaded by `avatar_gs.cli.preprocess_video`.
- Inputs: input video path, output frame directory, frame step, image quality.
- Outputs/downstream: controls extraction of `data/frames/input/*.jpg` and `metadata.json`.

### `configs/stage1_smplestx.yaml`
- Role: stage1 path and asset configuration.
- Interface: loaded by `avatar_gs.cli.stage1_fit_smplestx`.
- Inputs: frame directory, asset roots, SMPLest-X root, output directory.
- Outputs/downstream: determines where stage1 writes sequence assets; the stock `out_dir` is `outputs/stage1`.

### `configs/stage2_gaussian.yaml`
- Role: main stage2 configuration for frame selection, binding, supervision, losses, shell logic, and render defaults.
- Interface: loaded by `avatar_gs.cli.stage2_train_gaussian` and reused by `avatar_gs.cli.render`.
- Inputs: stage1 output paths, training parameters, supervision policy, shell settings, render settings.
- Outputs/downstream: defines default stage2 behavior.
- Critical fact: the current default still says `training_mode: inner_only`.
- Latest update: the stock config is now split across stage1 roots. Geometry and mask paths still point to `outputs/stage1_sharedbeta`, while `smpl_face_semantics_npz` points to `outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz`; it also relaxes `stage1_align_filter_min` to `0.93`, enables `inner_rgb_from_combined_rgb`, and raises several densify or local-prior candidate ratios.

## Scripts

### `scripts/auto_full_pipeline.ps1`
- Role: local orchestration script that runs stage1, stage2, and render in sequence and compares runs.
- Interface: PowerShell script; helper functions include run discovery and logging.
- Inputs: repository root, config path, stage1/stage2/render commands.
- Outputs/downstream: writes automation logs and preserves selected outputs.
- Warning: the name suggests a full pipeline, but the default config it calls is still `inner_only`.

### `scripts/export_gaussians_to_ply.py`
- Role: export stage2 checkpoints to PLY for inspection in external tools.
- Interface: `main()` plus helpers such as checkpoint loading and PLY writers.
- Inputs: stage2 checkpoint, optional mesh index, export mode.
- Outputs/downstream: PLY files for debugging or visualization.

## Evaluation Entrypoints

### `evaluation/stage1_eval/person_contour.py`
- Role: stage1 contour and person-mask helper layer.
- Interface: functions for loading masks, running MODNet-style evaluation masks, and extracting contours.
- Inputs: image frames, stored mask NPZ files, optional MODNet runtime.
- Outputs/downstream: contour and mask inputs for `run_stage1_eval.py`.

### `evaluation/stage1_eval/run_stage1_eval.py`
- Role: compare projected stage1 SMPL silhouettes against person masks or contours.
- Interface: `main()` plus per-frame evaluation helpers.
- Inputs: stage1 cameras, stage1 vertices or OBJ sequence, evaluation masks or contours.
- Outputs/downstream: JSON metrics and debug images under `evaluation/output_images/stage1`.

### `evaluation/stage1_eval/stage1_align_check.py`
- Role: compatibility wrapper around the stage1 evaluation path.
- Interface: `main()`.
- Inputs: command-line arguments.
- Outputs/downstream: forwards into the stage1 evaluation flow.

## `src/avatar_gs` Package Root and Export Layers

### `src/avatar_gs/__init__.py`
- Role: package marker only.
- Interface: none.
- Inputs: none.
- Outputs/downstream: allows `avatar_gs` imports.

### `src/avatar_gs/alignment/__init__.py`
- Role: alignment package export shell.
- Interface: package import layer.
- Inputs: alignment submodules.
- Outputs/downstream: stable import path for alignment utilities.

### `src/avatar_gs/deform/__init__.py`
- Role: deform package export shell.
- Interface: package import layer.
- Inputs: deform submodules.
- Outputs/downstream: stable import path for deform utilities.

### `src/avatar_gs/io/__init__.py`
- Role: export mesh I/O helpers.
- Interface: package import layer.
- Inputs: `mesh_export.py`.
- Outputs/downstream: used by stage1 mesh export paths.

### `src/avatar_gs/losses/__init__.py`
- Role: export robust and regularization losses.
- Interface: package import layer.
- Inputs: local loss modules.
- Outputs/downstream: consumed by stage2 training.

### `src/avatar_gs/segmentation/__init__.py`
- Role: export segmentation helpers.
- Interface: package import layer.
- Inputs: `modnet.py`, `fashn.py`, and face-semantics helper exports.
- Outputs/downstream: unified import path for mask estimation and FASHN-driven stage1 semantics export.

### `src/avatar_gs/stage2/__init__.py`
- Role: export stage2 config, supervision, mesh runtime, reporting, and rigid-align symbols.
- Interface: package import layer.
- Inputs: stage2 submodules.
- Outputs/downstream: shared import surface for CLI and training code.

### `src/avatar_gs/training/__init__.py`
- Role: export frame weighting and inner/outer supervision helpers.
- Interface: package import layer.
- Inputs: training helper submodules.
- Outputs/downstream: used by `train_loop.py`.

## CLI Entrypoints

### `src/avatar_gs/cli/run_all.py`
- Role: simple sequential orchestrator.
- Interface: `run(cmd)`, `main()`.
- Inputs: hard-coded preprocess, stage1, stage2, and render commands.
- Outputs/downstream: launches the full local chain in order.

### `src/avatar_gs/cli/preprocess_video.py`
- Role: frame extraction from the input video.
- Interface: `main()`.
- Inputs: `configs/video.yaml` plus CLI overrides and input video.
- Outputs/downstream: frame JPEGs, metadata, resolved YAML, and stage1-ready frame data.

### `src/avatar_gs/cli/_run_smplestx_refine_frames.py`
- Role: prepare and invoke SMPLest-X on selected frames.
- Interface: `main()` plus helpers for frame listing, index parsing, safe copying, model-name resolution, and checkpoint checks.
- Inputs: frame directory, selected frame indices, SMPLest-X repo path, pretrained model path.
- Outputs/downstream: a prepared runtime directory for `third_party/SMPLest-X/main/inference.py`.

### `src/avatar_gs/cli/stage1_fit_smplestx.py`
- Role: stage1 master executor.
- Interface: `main()` plus a large set of helpers for requirement checks, mask estimation, projection, parameter reconstruction, frame-record building, and export.
- Inputs: frame images, SMPLest-X outputs, MODNet outputs, stage1 config, body-model assets, and optional local FASHN assets.
- Outputs/downstream: all stage1 sequence assets consumed by stage2 and render, raw backend artifacts under `<out_dir>/smplestx`, optional face-semantics assets under `<out_dir>/fashn_face_semantics`, and summary metadata under `<out_dir>/final`.
- Data flow: frames -> `_run_smplestx_refine_frames.py` -> SMPLest-X -> rebuilt cameras and meshes -> fused masks -> optional shared-beta refit -> silhouette refine -> optional FASHN face semantics export -> exported sequence package.
- Default behavior: `disable_shared_beta_refit=True`, `build_fashn_face_semantics=True`, and silhouette refine runs unless explicitly disabled.

### `src/avatar_gs/cli/stage1_bestof_merge.py`
- Role: merge best per-frame stage1 assets from multiple evaluated candidate runs.
- Interface: `main()` plus helpers for evaluation-report lookup, per-frame scoring, and NPZ merging.
- Inputs: candidate stage1 directories, a base stage1 directory, and `evaluation/output_images/**/stage1_eval_report_all.json`.
- Outputs/downstream: writes `outputs/stage1_bestof/**`, copies per-frame OBJ files, and records source provenance in `smpl_params_sequence.npz`.

### `src/avatar_gs/cli/stage1_align_check.py`
- Role: CLI entry for stage1 alignment checks.
- Interface: `main()`.
- Inputs: CLI arguments.
- Outputs/downstream: forwards to silhouette-alignment evaluation logic.

### `src/avatar_gs/cli/stage2_train_gaussian.py`
- Role: stage2 launcher and coordinator.
- Interface: `main()` plus helpers for project-path resolution, stage1 frame filtering, alignment-score loading, camera-tensor alignment, and optional init-checkpoint loading.
- Inputs: stage2 config, stage1 assets, optional init checkpoint, and optionally the stage1 face-semantics NPZ referenced by config.
- Outputs/downstream: trained Gaussian checkpoints, supervision cache or summaries, outer reports.
- Data flow: config -> frame selection -> rigid align -> init gaussians -> supervision state -> semantic asset resolution -> mesh runtime -> train loop.

### `src/avatar_gs/cli/render.py`
- Role: render executor for real, foreground-only, and deform views.
- Interface: `main()` plus helpers for frame selection, checkpoint loading, pose-drive selection, layer splitting, image I/O, overlays, deform-mesh fallback, and output-directory layout.
- Inputs: stage2 checkpoint, stage1 meshes or params, original images, render config.
- Outputs/downstream: rendered PNG images across combined, inner, and outer shell views.
- Data flow: checkpoint + selected mesh/pose source -> per-frame Gaussian selection -> renderer -> optional overlays -> saved images.
- Latest update: when driven by the stock config, render inherits the same split-root stage1 defaults as stage2, with geometry paths still targeting `outputs/stage1_sharedbeta`.

### `src/avatar_gs/cli/edit.py`
- Role: small edit utility over saved stage2 checkpoints.
- Interface: `main()` and `_apply_simple_edit`.
- Inputs: a saved Gaussian checkpoint and edit parameters.
- Outputs/downstream: an edited checkpoint for later render or experimentation.

## Alignment Modules

### `src/avatar_gs/alignment/silhouette.py`
- Role: mask and silhouette geometry utilities.
- Interface: helpers for bounding boxes, IoU, similarity transforms, UV-based alignment, and camera-point transforms.
- Inputs: mask images, UV coordinates, mesh faces, camera-space points.
- Outputs/downstream: alignment metrics and transforms used by stage1 or stage2 alignment logic.

### `src/avatar_gs/alignment/mask_sources.py`
- Role: unified mask-source resolution layer.
- Interface: projection helpers, rasterizers, mask cleaning and plausibility filters, stage1 camera readout, mask loading, chroma-key estimation, MODNet evaluation helpers, and source resolution functions.
- Inputs: images, vertices, faces, camera parameters, NPZ mask caches.
- Outputs/downstream: person masks used by stage1, stage2, and evaluation.

## Data Modules

### `src/avatar_gs/data/video_reader.py`
- Role: OpenCV video reader wrapper.
- Interface: `VideoReader`.
- Inputs: video path.
- Outputs/downstream: indexed BGR frames for preprocess.

### `src/avatar_gs/data/frame_dataset.py`
- Role: PyTorch dataset over extracted RGB frames.
- Interface: `FrameDataset`, local RGB-read helper.
- Inputs: frame directory and `metadata.json`.
- Outputs/downstream: frame tensors for stage1.

### `src/avatar_gs/data/keypoints_mediapipe.py`
- Role: body and hand keypoint extraction with MediaPipe.
- Interface: `extract_keypoints_from_frames`.
- Inputs: frame images.
- Outputs/downstream: keypoints and bounding-box hints used in stage1.

### `src/avatar_gs/data/sequence_mesh.py`
- Role: loader for stage1 OBJ mesh sequences.
- Interface: `SequenceMeshProvider` and frame-index parsing helpers.
- Inputs: `outputs/stage1/sequence/frame_*.obj`.
- Outputs/downstream: frame-indexed meshes for stage2 and render.

### `src/avatar_gs/data/camera_utils.py`
- Role: placeholder shell.
- Interface: none of substance.
- Inputs: none.
- Outputs/downstream: mostly misleading file presence.

### `src/avatar_gs/data/mask_utils.py`
- Role: placeholder shell.
- Interface: none of substance.
- Inputs: none.
- Outputs/downstream: mostly misleading file presence.

## Deform Module

### `src/avatar_gs/deform/smplx_param_source.py`
- Role: rebuild deform-time vertices from saved SMPL parameter sequences.
- Interface: loader helpers for frame-index or frame-name lookup.
- Inputs: `smpl_params_sequence.npz` and frame selection.
- Outputs/downstream: deform mesh vertices for render.

## I/O Module

### `src/avatar_gs/io/mesh_export.py`
- Role: OBJ mesh export utilities.
- Interface: face-text builder and `export_obj`.
- Inputs: mesh vertices and faces.
- Outputs/downstream: OBJ sequence files written during stage1.

## Loss Modules

### `src/avatar_gs/losses/robust.py`
- Role: robust photometric losses.
- Interface: Huber, Charbonnier, and wrapper loss functions.
- Inputs: predicted and target tensors.
- Outputs/downstream: scalar losses consumed by stage2 training.

### `src/avatar_gs/losses/regularizers.py`
- Role: Gaussian regularization terms.
- Interface: opacity prior, covariance penalties, canonical offset penalties, outer color-band penalties, and layer balance losses.
- Inputs: Gaussian state tensors.
- Outputs/downstream: regularization terms added inside the training loop.

## Segmentation Module

### `src/avatar_gs/segmentation/modnet.py`
- Role: MODNet runtime adapter for person-mask estimation.
- Interface: checkpoint resolution, mask-estimation functions, and runtime wrapper state.
- Inputs: RGB images and MODNet checkpoint path.
- Outputs/downstream: person alpha or mask tensors for stage1, stage2, and evaluation.

## SMPL Placeholder Modules

### `src/avatar_gs/smpl/lbs.py`
- Role: dummy linear blend skinning placeholder.
- Interface: `linear_blend_skinning`.
- Inputs: pose-like tensors.
- Outputs/downstream: placeholder output only; not a trusted implementation.

### `src/avatar_gs/smpl/smpl_model.py`
- Role: dummy SMPL wrapper placeholder.
- Interface: `DummySMPL`.
- Inputs: none of production value.
- Outputs/downstream: placeholder behavior; should not be mistaken for the real model layer.

## Gaussian Core

### `src/avatar_gs/gaussian/structured_gaussian.py`
- Role: main Gaussian state container.
- Interface: `Gaussian3D`.
- Inputs: Gaussian tensors and metadata.
- Outputs/downstream: checkpoint state used by training and render.
- Notes: stores more than positions and colors; it carries shell, binding, quality-gate, and pruning metadata.

### `src/avatar_gs/gaussian/init_from_smpl.py`
- Role: initialize Gaussians from mesh geometry.
- Interface: OBJ loading helpers and Gaussian-init functions.
- Inputs: initial mesh vertices and faces.
- Outputs/downstream: initial `Gaussian3D` state for stage2.

### `src/avatar_gs/gaussian/losses.py`
- Role: simple Gaussian-space losses.
- Interface: basic tensor loss helpers.
- Inputs: training tensors.
- Outputs/downstream: additional loss terms inside stage2.

### `src/avatar_gs/gaussian/rigid_align.py`
- Role: rigid-transform geometry utilities.
- Interface: Euler-to-rotation conversion, rigid transform application, coordinate-descent rigid search.
- Inputs: points, masks, rotation or translation candidates.
- Outputs/downstream: rigid-align helpers used by stage2 and render.

### `src/avatar_gs/gaussian/shell_policy.py`
- Role: shell budget, scoring, spawn, soft-freeze, hard-prune, and outer-prune policy.
- Interface: shell-count estimators and candidate selectors.
- Inputs: Gaussian statistics, support scores, shell metadata.
- Outputs/downstream: dynamic shell-management decisions used during training.

### `src/avatar_gs/gaussian/train_loop.py`
- Role: main stage2 optimization loop.
- Interface: public training entry plus internal implementation helpers.
- Inputs: `Gaussian3D`, stage2 config, supervision state, mesh runtime, binding state, optimizer state.
- Outputs/downstream: updated checkpoints, logs, shell-policy state, and metrics.
- Data flow per iteration: select frame -> update binding -> render -> compute losses -> optimize -> densify/prune -> log and checkpoint.
- Latest update: semantic-aware policies now use face-level semantics for inner local color donor selection and exposed-face blocking in outer logic.
- Warning: this file is overloaded and is the main maintenance risk.

## Gaussian Binding

### `src/avatar_gs/gaussian/binding.py`
- Role: shared binding API layer.
- Interface: binding entrypoints and shared helpers.
- Inputs: binding config, mesh-frame data, Gaussian state.
- Outputs/downstream: used by inner and outer binding layers.

### `src/avatar_gs/gaussian/binding_layers/__init__.py`
- Role: package shell for binding layers.
- Interface: import surface.
- Inputs: binding-layer submodules.
- Outputs/downstream: stable import path.

### `src/avatar_gs/gaussian/binding_layers/types.py`
- Role: type and config definitions for binding.
- Interface: binding mode and per-frame data structures.
- Inputs: config values and mesh-derived data.
- Outputs/downstream: contracts shared by binding builders and updates.

### `src/avatar_gs/gaussian/binding_layers/frame_builder.py`
- Role: construct per-frame binding geometry.
- Interface: frame-data builders and binding-mode parsing/export helpers.
- Inputs: mesh faces, vertices, binding settings.
- Outputs/downstream: `BindingFrameData` consumed by inner and outer updates.

### `src/avatar_gs/gaussian/binding_layers/binding_update.py`
- Role: binding update math.
- Interface: quaternion helpers, interpolation helpers, temporal stabilization, and target builders.
- Inputs: rotation matrices, barycentric geometry, temporal sequences.
- Outputs/downstream: per-frame targets for Gaussian binding.

### `src/avatar_gs/gaussian/binding_layers/inner_binding.py`
- Role: inner-layer binding logic.
- Interface: bind-mask computation and per-frame application helpers.
- Inputs: `Gaussian3D`, `BindingFrameData`, inner-binding config.
- Outputs/downstream: updated inner Gaussian transforms.

### `src/avatar_gs/gaussian/binding_layers/outer_binding.py`
- Role: outer-layer binding logic.
- Interface: anchor-mask computation and per-frame outer-anchor application helpers.
- Inputs: `Gaussian3D`, `BindingFrameData`, face anchors, local offsets.
- Outputs/downstream: updated outer Gaussian transforms.

## Gaussian Densify and Visibility

### `src/avatar_gs/gaussian/densify/__init__.py`
- Role: package shell for densify logic.
- Interface: import layer.
- Inputs: densify submodules.
- Outputs/downstream: stable import path.

### `src/avatar_gs/gaussian/densify/stats.py`
- Role: hold running statistics used to decide densification.
- Interface: `GaussianDensifyStats`.
- Inputs: visibility, residual, support, or training signals.
- Outputs/downstream: statistics queried by densify rules.

### `src/avatar_gs/gaussian/densify/rules.py`
- Role: update densify stats and select densify candidates.
- Interface: rule functions for stats updates and candidate selection.
- Inputs: current Gaussian state and collected statistics.
- Outputs/downstream: densification decisions inside training.

### `src/avatar_gs/gaussian/visibility/__init__.py`
- Role: package shell for visibility logic.
- Interface: import layer.
- Inputs: visibility submodules.
- Outputs/downstream: stable import path.

### `src/avatar_gs/gaussian/visibility/gaussian_visibility.py`
- Role: estimate Gaussian visibility against cameras and masks.
- Interface: visibility computation functions.
- Inputs: camera intrinsics or extrinsics, Gaussian state, alpha or person masks.
- Outputs/downstream: visibility scores used by shell policy and pruning.

### `src/avatar_gs/gaussian/visibility/border_attenuation.py`
- Role: attenuate loss or support near borders.
- Interface: border-weight builders and samplers.
- Inputs: mask borders and UV or image-space coordinates.
- Outputs/downstream: weights used in photometric and regularization logic.

## Renderer

### `src/avatar_gs/renderer/gaussian_renderer.py`
- Role: wrapper over `gsplat` rendering.
- Interface: `render_gaussians_3d`.
- Inputs: `Gaussian3D`, camera parameters, render settings.
- Outputs/downstream: RGB, alpha, and render metadata consumed by training and render CLI.

### `src/avatar_gs/renderer/camera.py`
- Role: placeholder shell.
- Interface: no substantial contract.
- Inputs: none.
- Outputs/downstream: mostly confusion.

## Stage2 Support Modules

### `src/avatar_gs/stage2/config.py`
- Role: parse and normalize stage2 config sections.
- Interface: dataclasses and config objects for frame selection, binding, supervision, and training.
- Inputs: YAML config plus CLI overrides.
- Outputs/downstream: typed config used across stage2 launcher, mesh runtime, supervision, and train loop.
- Latest update: resolves `smpl_face_semantics_npz` and semantic-policy thresholds into the stage2 runtime config.

### `src/avatar_gs/stage2/mesh_runtime.py`
- Role: build runtime mesh context for stage2 and render.
- Interface: mesh-path resolvers, bind-index resolution, context classes, runtime classes, and binding-state initializers.
- Inputs: stage1 OBJ sequence, stage1 vertex NPZ, frame lists, rigid-align config.
- Outputs/downstream: per-frame mesh context and binding state used by stage2 training and render.

### `src/avatar_gs/stage2/rigid_align.py`
- Role: stage2 rigid-align search controller.
- Interface: config normalization, search functions, and result dataclasses.
- Inputs: stage1 mesh or silhouette data, person masks, search settings.
- Outputs/downstream: rigid alignment parameters consumed by mesh runtime and render.

### `src/avatar_gs/stage2/supervision.py`
- Role: build and cache stage2 supervision assets.
- Interface: frame-asset structures, cache loaders, supervision-state builders, and preparation helpers.
- Inputs: stage1 masks, images, mesh projections, MODNet outputs, frame-selection data.
- Outputs/downstream: RGB targets, person masks, contour masks, frame weights, and cache artifacts for training.

### `src/avatar_gs/stage2/reporting.py`
- Role: stage2 reporting and outer-quality analysis.
- Interface: threshold builders, path preparation, metric evaluation, JSON or JSONL writers, outer-face records, and spawn reports.
- Inputs: support scores, shell quality signals, outside ratios, spawn decisions.
- Outputs/downstream: quality metrics and analysis reports under `outputs/stage2`.

## Training Helpers

### `src/avatar_gs/training/frame_weighting.py`
- Role: residual-based frame weighting.
- Interface: `FrameResidualEMA` and frame-weight helper functions.
- Inputs: per-frame residual history.
- Outputs/downstream: weights used to bias stage2 training.

### `src/avatar_gs/training/inner_layer_training.py`
- Role: resolve inner-layer supervision masks.
- Interface: inner supervision mask helpers.
- Inputs: SMPL masks, person masks, config values.
- Outputs/downstream: mask tensors used by inner loss terms.

### `src/avatar_gs/training/outer_layer_training.py`
- Role: outer trigger-ring construction and face-gap target estimation.
- Interface: trigger-mask building, trigger-to-face mapping, and face-gap target estimation.
- Inputs: person contour or mask, SMPL masks, mesh-face geometry.
- Outputs/downstream: outer supervision masks and face targets used by the training loop.

## Utilities

### `src/avatar_gs/utils/cfg.py`
- Role: YAML config loading, overrides, path resolution, and saving.
- Interface: config read, deep-update, override parsing, and save helpers.
- Inputs: YAML files and CLI override strings.
- Outputs/downstream: resolved config dictionaries for every CLI entry.

### `src/avatar_gs/utils/io.py`
- Role: small filesystem and JSON helpers.
- Interface: directory creation and JSON read or write functions.
- Inputs: file paths and Python data.
- Outputs/downstream: shared utility for stage1, stage2, and render.

### `src/avatar_gs/utils/log.py`
- Role: logger setup utility.
- Interface: `setup_logger`.
- Inputs: logger name and output configuration.
- Outputs/downstream: consistent CLI logging.

### `src/avatar_gs/utils/progress.py`
- Role: small progress-wrapper compatibility layer.
- Interface: exported `tqdm`.
- Inputs: iterable work units and optional progress labels.
- Outputs/downstream: shared progress bars for stage1 CLI and related helpers.

### `src/avatar_gs/utils/person_contour.py`
- Role: shim that re-exports contour helpers from evaluation code.
- Interface: imported contour utilities.
- Inputs: evaluation contour module.
- Outputs/downstream: avoids duplicated contour logic.

### `src/avatar_gs/utils/seed.py`
- Role: deterministic seed setup.
- Interface: `seed_everything`.
- Inputs: integer seed values.
- Outputs/downstream: reproducible PyTorch and NumPy setup.

## Vendored `src/lib` Modules

### `src/lib/__init__.py`
- Role: package marker for vendored motion-prior components.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/THIRD_PARTY_LICENSE_STAGE1_MOTION_PRIOR.txt`
- Role: third-party license record for vendored stage1 motion-prior code.
- Interface: text file.
- Inputs: upstream license text.
- Outputs/downstream: legal reference only.

### `src/lib/core/__init__.py`
- Role: package marker.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/core/config.py`
- Role: configuration defaults for the vendored motion-prior stack.
- Interface: config helpers, including default data-root resolution.
- Inputs: local environment and expected data paths.
- Outputs/downstream: motion-prior config used during stage1-related processing.

### `src/lib/data_utils/__init__.py`
- Role: package marker.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/data_utils/img_utils.py`
- Role: image crop and patch utilities.
- Interface: crop functions, patch generation, demo image crop helpers, and chunk splitting.
- Inputs: images, bounding boxes, patch parameters.
- Outputs/downstream: normalized crops for motion-prior or SPIN-style processing.

### `src/lib/dataset/__init__.py`
- Role: package marker.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/dataset/inference.py`
- Role: inference dataset wrappers for motion-prior or SPIN-style processing.
- Interface: `Inference`, `ImageFolder`.
- Inputs: image folders, joints, and bounding-box hints.
- Outputs/downstream: dataset samples for stage1 auxiliary processing.

### `src/lib/models/__init__.py`
- Role: package marker.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/models/motion_prior.py`
- Role: GRU-based motion-prior network definitions.
- Interface: temporal encoder, motion-prior network, demo wrappers.
- Inputs: pose sequences and temporal features.
- Outputs/downstream: smoothed or prior-conditioned pose predictions.

### `src/lib/models/smpl.py`
- Role: local wrapper around `smplx.SMPL` plus face helpers.
- Interface: `SMPL`, `get_smpl_faces`.
- Inputs: SMPL parameters.
- Outputs/downstream: SMPL meshes and face data for auxiliary stage1 logic.

### `src/lib/models/spin.py`
- Role: SPIN/HMR-style image-to-body modules.
- Interface: `HMR`, `Regressor`, factory functions, projection utilities.
- Inputs: image crops or features.
- Outputs/downstream: pose, shape, and camera estimates used by vendored stage1 support code.

### `src/lib/utils/__init__.py`
- Role: package marker.
- Interface: none.
- Inputs: none.
- Outputs/downstream: import support.

### `src/lib/utils/geometry.py`
- Role: body-geometry math utilities.
- Interface: Rodrigues conversion, quaternion helpers, 6D rotation conversion, translation estimation, projection-related helpers.
- Inputs: rotations, joints, vertices, camera-space signals.
- Outputs/downstream: geometric transforms used by vendored stage1 support code.

### `src/lib/utils/one_euro_filter.py`
- Role: temporal smoothing filter.
- Interface: `OneEuroFilter`.
- Inputs: time-series signals.
- Outputs/downstream: smoothed pose or motion values.

### `src/lib/utils/smooth_bbox.py`
- Role: bounding-box smoothing.
- Interface: bbox smoothing and bbox-from-keypoints helpers.
- Inputs: per-frame boxes or keypoints.
- Outputs/downstream: stable crop boxes.

### `src/lib/utils/smooth_pose.py`
- Role: pose smoothing using One Euro filtering over SMPL-related parameters.
- Interface: `smooth_pose`.
- Inputs: pose, betas, joints, vertices.
- Outputs/downstream: smoothed pose sequences used by stage1 support paths.

## Data and Output Contracts

### Primary inputs
- `data/videos/input.mp4`
- `data/frames/input/*.jpg`
- `data/frames/input/metadata.json`

### Stage1 outputs
- `outputs/stage1/cameras.json`
- `outputs/stage1/keypoints.json`
- `outputs/stage1/person_masks.npz`
- `outputs/stage1/smpl_params_sequence.npz`
- `outputs/stage1/smpl_vertices_sequence.npz`
- `outputs/stage1/smpl_joints_sequence.npz`
- `outputs/stage1/smpl_skeleton_sequence.npz`
- `outputs/stage1/sequence/frame_*.obj`
- `outputs/stage1/smplestx/**`
- `outputs/stage1/final/run_summary.json`
- `outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz`
- `outputs/stage1/fashn_face_semantics/smpl_face_semantics.csv`
- `outputs/stage1/fashn_face_semantics/smpl_face_semantics_summary.json`
- `outputs/stage1_bestof/**`
- `outputs/stage1_sharedbeta/**` as an alternate or legacy root still referenced by the stock stage2 config

### Stage2 outputs
- `outputs/stage2/gaussians3d.pt`
- `outputs/stage2/gaussians3d_inner.pt`
- `outputs/stage2/supervision_cache.npz`
- `outputs/stage2/supervision_summary.json`
- `outputs/stage2/outer_quality_metrics.jsonl`
- `outputs/stage2/outer_quality_report.json`
- `outputs/stage2/outer_spawn_per_frame.json`
- `outputs/stage2/supervision_preview/**`

### Render outputs
- `outputs/renders/real/{combined,inner,outer}/*.png`
- `outputs/renders/fg_only/{combined,inner,outer}/*.png`
- `outputs/renders/deform/{combined,inner,outer}/*.png`

### Evaluation outputs
- `evaluation/output_images/stage1/*_gt_stage1_contour.png`
- `evaluation/output_images/stage1/*_stage1_eval_report.json`
- `evaluation/output_images/stage1/stage1_eval_report_all.json`

## External Boundaries

### `third_party/SMPLest-X/**`
- Boundary role: external body-estimation stack used by stage1.
- Entry points used here: primarily `third_party/SMPLest-X/main/inference.py` and editable install under `main/transformer_utils`.
- Contract with this repo: produce body predictions that `stage1_fit_smplestx.py` rebuilds into stage1 assets.

### `third_party/MODNet/**`
- Boundary role: external portrait matting stack.
- Entry points used here: called through `src/avatar_gs/segmentation/modnet.py`.
- Contract with this repo: produce person alpha or mask estimates for stage1, stage2, and evaluation.

### `outputs/**`
- Boundary role: generated artifacts, not source code.
- Contract with this repo: stage-to-stage handoff storage and debug output.
- Warning: generated files should not be confused with maintained logic.

## End-to-End Data Flow

1. `preprocess_video.py`
   `input.mp4 -> frame jpg + metadata.json`

2. `stage1_fit_smplestx.py`
   `frame jpg + SMPLest-X + body assets -> raw smplestx outputs + rebuilt stage1 assets + optional shared-beta or silhouette postprocess + optional face semantics`

3. `stage1_bestof_merge.py` (optional)
   `multiple evaluated stage1 dirs + reports -> best per-frame merged stage1 root`

4. `stage2_train_gaussian.py`
   `stage1 assets + stage2 config -> Gaussian3D + supervision state + mesh runtime -> trained checkpoints and reports`

5. `render.py`
   `trained checkpoint + stage1 mesh or params + frame images -> real/fg_only/deform renders`

6. `run_stage1_eval.py`
   `stage1 vertices or OBJ + cameras + person mask or contour -> stage1 alignment metrics and debug images`

## Maintenance Priority Files

If someone changes only a few files and still expects to understand the repo, these are the ones that matter most:
- `README.md`
- `requirements.txt`
- `idea.md`
- `PROJECT_MAP.md`
- `configs/stage2_gaussian.yaml`
- `src/avatar_gs/cli/stage1_fit_smplestx.py`
- `src/avatar_gs/cli/stage2_train_gaussian.py`
- `src/avatar_gs/cli/render.py`
- `src/avatar_gs/gaussian/train_loop.py`
- `src/avatar_gs/stage2/mesh_runtime.py`
- `src/avatar_gs/stage2/supervision.py`
- `src/avatar_gs/training/outer_layer_training.py`

If these files drift away from each other, the repository story becomes fiction again.
