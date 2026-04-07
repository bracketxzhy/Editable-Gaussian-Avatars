# idea

## Scope

This file records the uncomfortable truth about the repository: what it is actually doing, where the pipeline branches, and where the current naming or defaults are misleading.

Document split:
- `README.md`: external story, setup, reproduction, usage.
- `PROJECT_MAP.md`: file responsibilities, interfaces, and data flow.
- `idea.md`: reality check, branch points, risks, and maintenance priorities.

Naming rule:
- Stable docs should point to role-based evaluation roots such as `evaluation/eval_outputs/stage2/current_inner_body_baseline_20260406/`, not transient experiment tags.
- Experiment ids belong in run history, not in the repository's long-lived map.

If these three files disagree, trust the code first and fix the docs second.

## What This Repository Really Is

- This is not a standalone human reconstruction stack.
- It is a pipeline that wraps and organizes external systems: `SMPLest-X`, `MODNet`, `gsplat`, SMPL/SMPL-X body models, and external checkpoints.
- The intended product is an editable Gaussian avatar bound to a per-frame SMPL mesh sequence.
- The effective execution chain is:
  1. `src/avatar_gs/cli/preprocess_video.py`
  2. `src/avatar_gs/cli/stage1_fit_smplestx.py`
  3. `src/avatar_gs/cli/stage2_train_gaussian.py`
  4. `src/avatar_gs/cli/render.py`
  5. optional evaluation: `evaluation/stage1_eval/run_stage1_eval.py`

## The Main Pipeline

### Step 1: preprocess

- Input: `data/videos/input.mp4`
- Output: `data/frames/input/*.jpg`, `data/frames/input/metadata.json`
- Real role: frame extraction plus metadata capture. Nothing smart happens here.

### Step 2: stage1

- Entry: `src/avatar_gs/cli/stage1_fit_smplestx.py`
- Input: extracted frames, SMPLest-X repo, body-model assets, checkpoints, optional MODNet
- Output:
  - `outputs/stage1/cameras.json`
  - `outputs/stage1/keypoints.json`
  - `outputs/stage1/person_masks.npz`
  - `outputs/stage1/smpl_params_sequence.npz`
  - `outputs/stage1/smpl_vertices_sequence.npz`
  - `outputs/stage1/smpl_joints_sequence.npz`
  - `outputs/stage1/smpl_skeleton_sequence.npz`
  - `outputs/stage1/sequence/frame_*.obj`
  - `outputs/stage1/final/run_summary.json`
- Real role: not a homegrown body fitter; it is a SMPLest-X runner, result rebuilder, mask generator, and stage2 asset exporter.

### Step 3: stage2

- Entry: `src/avatar_gs/cli/stage2_train_gaussian.py`
- Input: stage1 assets, stage2 config, optional init checkpoint
- Output:
  - `outputs/stage2/gaussians3d.pt`
  - `outputs/stage2/gaussians3d_inner.pt`
  - `outputs/stage2/supervision_cache.npz`
  - `outputs/stage2/supervision_summary.json`
  - `outputs/stage2/outer_quality_metrics.jsonl`
  - `outputs/stage2/outer_quality_report.json`
  - `outputs/stage2/outer_spawn_per_frame.json`
  - `outputs/stage2/supervision_preview/**`
- Real role: build and train Gaussian layers that are attached to the SMPL sequence.

### Step 4: render

- Entry: `src/avatar_gs/cli/render.py`
- Input: stage2 checkpoint, stage1 mesh or params, frame images
- Output: `outputs/renders/real|fg_only|deform/{combined,inner,outer}/*.png`
- Real role: not a passive viewer. It is another executor that chooses pose source, mesh source, shell split, overlays, and output modes.

## Current Stage1 Update: Silhouette Refine, Optional Shared-Beta, and FASHN

The current stage1 implementation adds a second control axis on top of the old geometry-plus-mask story.

Before this stage1 update, the repository was mainly:
- geometry
- person masks
- shell policies

After this update, the repository also carries:
- default-on silhouette-driven per-frame refinement
- sequence-level shared body-shape refit that is optional and disabled by default
- face-level semantics inferred from local FASHN parsing
- optional best-of merge across evaluated stage1 candidates

The important implication is that stage1 is no longer just a body-parameter exporter.
It is now also a postprocess stage and an optional semantics-export stage.

## Where the Logic Branches

### Branch 1: stage1 name versus stage1 reality

The file name suggests a local SMPL fitting stage. That is not what the code is centered on.

What it really does:
- prepare SMPLest-X runtime and checkpoints
- call `src/avatar_gs/cli/_run_smplestx_refine_frames.py`
- rebuild camera, joints, and vertices from SMPLest-X outputs
- estimate and fuse person masks
- optionally run shared-beta refit
- run silhouette-driven parameter refinement unless explicitly disabled
- optionally export FASHN face semantics
- export the assets required by stage2

Conclusion:
- stage1 quality is dominated by `third_party/SMPLest-X` and external assets
- pretending it is a self-contained in-repo optimizer is fiction

### Branch 2: person-mask supervision has multiple sources

Mask supervision is not a single truth source.

Stage1 can use:
- chroma-key mask estimation
- MODNet matting
- SMPL projection as a geometric hint

Stage2 can use:
- stored `outputs/stage1/person_masks.npz`
- a fresh MODNet pass, depending on config
- fusion logic controlled by config

Evaluation can use:
- stored masks
- fresh contour extraction
- fresh MODNet-based evaluation paths

Consequence:
- many failures that look geometric are actually supervision-source mismatches
- if mask provenance is not controlled, metrics are not trustworthy

### Branch 3: mesh source also branches

Stage2 and render can pull mesh data from different places:
- `outputs/stage1/sequence/frame_*.obj`
- `outputs/stage1/smpl_vertices_sequence.npz`
- deform-time mesh reconstruction in render

`src/avatar_gs/stage2/mesh_runtime.py` exists largely to normalize those sources into a consistent runtime mesh context.

### Branch 4: inner and outer are already two systems

Inner and outer Gaussians are not the same thing with one extra switch.

Inner:
- body-attached
- stable
- conservative
- mostly vertex-driven binding

Outer:
- shell or residue compensation
- hair, clothing edge, detached contour behavior
- mostly face-anchor driven binding with local offsets

Calling this a single Gaussian model is inaccurate. It is two behavior systems sharing a training loop.

### Branch 5: outer supervision is now trigger-ring driven

The old residual-only story is obsolete.

Current outer supervision centers on `src/avatar_gs/training/outer_layer_training.py`, especially trigger-ring logic:
- derive `outer_base_mask` from GT/person contour signals
- subtract a dilated SMPL region
- build a trigger ring
- optionally gate by edges
- fall back when the support area is too small
- map image-space signals to face-level activation and gap targets

That means the correct description is:
- contour-driven local shell supervision
- not generic free-floating residual cleanup

### Branch 6: code capability versus default execution mode

This is the biggest current disconnect.

- The codebase, naming, and parameter surface clearly describe a dual-layer system.
- But `configs/stage2_gaussian.yaml` still defaults to `training_mode: inner_only`.
- `scripts/auto_full_pipeline.ps1` sounds like a full pipeline, yet it still routes through that default config.

So the default run is not a full dual-layer training pass.
It is an inner-only path.

If documentation or scripts still imply otherwise, they are lying.

### Branch 6.1: inner training logic still contains legacy and fallback branches

Do not pretend the inner-stage logic is now a clean-room minimal implementation.

Current code still contains:
- legacy binding modes in `src/avatar_gs/gaussian/binding_layers/inner_binding.py` and `frame_builder.py`
- compatibility switches such as `inner_legacy_e526_compat_enable` in `src/avatar_gs/gaussian/train_loop.py`
- fallback branches in mesh/runtime and outer-gap handling

That means the code is functional, but not yet stripped of historical baggage.
If someone claims the inner training path is already free of compatibility or fallback logic, they are either guessing or lying.

### Branch 7: supervision can be online or cached

`src/avatar_gs/stage2/supervision.py` supports two modes:
- compute supervision online
- precompute with `--prepare_supervision_only` and load `supervision_cache.npz`

This is useful, but it introduces cache drift risk.
A stale cache can make training look stable while consuming outdated supervision.

### Branch 8: rigid alignment is not optional cleanup anymore

- `src/avatar_gs/stage2/rigid_align.py` searches rigid transforms
- `src/avatar_gs/gaussian/rigid_align.py` provides the geometry utilities
- `src/avatar_gs/stage2/mesh_runtime.py` and `src/avatar_gs/cli/render.py` continue to apply the result

The repository has already admitted the truth:
- stage1 meshes are not reliably aligned to the real frames
- rigid correction is part of the pipeline, not cosmetic cleanup

### Branch 9: render has grown into a second executor

`src/avatar_gs/cli/render.py` now handles:
- frame or pose-drive selection
- deform mesh fallback logic
- layer splitting
- six-view rendering
- skeleton and keypoint overlays
- debug visibility and bound-only modes

That matters because render choices affect how users interpret model quality.
It is not a neutral output stage.

### Branch 10: placeholder and dummy files still exist

These files create false expectations:
- `src/avatar_gs/__init__.py`
- `src/avatar_gs/data/camera_utils.py`
- `src/avatar_gs/data/mask_utils.py`
- `src/avatar_gs/renderer/camera.py`
- `src/avatar_gs/smpl/lbs.py`
- `src/avatar_gs/smpl/smpl_model.py`

The first group is mostly empty shell code.
The second group is dummy implementation.

They are dangerous because the file names look official while the contents are not.

## Reproduction Reality

The project is not reproducible by `pip install -r requirements.txt` alone.

A real reproduction requires:
- a CUDA/PyTorch-matched install of `torch` and `torchvision`
- `requirements.txt`
- separate install of `mmcv-full==1.7.1`
- editable install of `third_party/SMPLest-X/main/transformer_utils`
- SMPLest-X checkpoints
- MODNet checkpoints
- SMPL / SMPL-X body-model assets

That is why the docs must keep Python dependencies separate from external assets and model files.
Mixing them together is lazy and misleading.

### Branch 11: stage1 output provenance is now split in several incompatible ways

The stage1 CLI default, the optional shared-beta branch, the optional best-of branch, and the stock stage2 config no longer point to one canonical root.

Current reality:
- `stage1_fit_smplestx.py` writes to `outputs/stage1` by default
- shared-beta refit rewrites records in the same output dir only if `--enable_shared_beta_refit` is passed
- `stage1_bestof_merge.py` can build a separate `outputs/stage1_bestof`
- `configs/stage2_gaussian.yaml` still points `stage1_dir` and `person_masks_npz` at `outputs/stage1_sharedbeta`
- the same config points `smpl_face_semantics_npz` at `outputs/stage1/fashn_face_semantics/smpl_face_semantics.npz`

If downstream code or documentation does not say which branch it consumes, the result is not just ambiguous.
It is operationally broken.

### Branch 12: semantics are now a second supervision-control path

Mask logic still answers where the person and silhouette support are.
The new face-semantics asset answers something different:
- whether a face is exposed body
- whether a face is covered by clothing or hair
- whether the label is too uncertain to trust

That semantic branch now changes stage2 behavior through:
- inner local color donor selection
- exposed-face exclusion from outer-shell logic

### Branch 13: richer outer-shell logic still does not make the default run full

The code is now smarter about where outer behavior should not happen.
But the default config still says `training_mode: inner_only`.

So the repository got semantically smarter without fixing the oldest honesty problem.
The default run is still not the same thing as a full dual-layer pipeline.

## Current Risks

1. The default config, stage1 defaults, and the repository narrative do not match.
   The code advertises dual-layer behavior, the default run remains `inner_only`, and the stock stage2 config still points at `outputs/stage1_sharedbeta` even though stage1 writes `outputs/stage1` and disables shared-beta by default.

2. Mask provenance dominates downstream quality.
   Alignment, trigger-ring activation, spawn/prune logic, and evaluation all depend on it.

3. `src/avatar_gs/gaussian/train_loop.py` and `src/avatar_gs/cli/render.py` are too heavy.
   They are turning into change-hazard files.

4. Supervision caches can drift away from the config that produced them.
   Without explicit validation, cached supervision is an integrity risk.

5. Placeholder and dummy modules keep polluting reader expectations.

6. Documentation can split again.
   `README.md`, `PROJECT_MAP.md`, and `idea.md` must be updated together when defaults, data contracts, or reproduction steps change.

7. Vendored runtime bulk now hides logic changes.
   The FASHN runtime tree makes it much easier for meaningful source changes to disappear inside a giant diff.

8. Semantic assets introduce a topology-mismatch failure mode.
   A stale `smpl_face_semantics.npz` can be structurally valid as a file but still invalid for the current face topology or stage1 sequence.

## What the Owner Should Do Next

1. Decide what the default product really is.
   If the goal is a true dual-layer avatar, stop shipping `inner_only` as the quiet default.

2. Split `src/avatar_gs/gaussian/train_loop.py`.
   At minimum separate supervision, binding, losses, shell policy, and checkpoint/reporting logic.

3. Unify mask policy.
   Stage1, stage2, and evaluation should not each tell a different story about mask precedence.

4. Rename or remove placeholder and dummy files.
   Keep them only if they are clearly labeled as placeholders.

5. Fix `scripts/auto_full_pipeline.ps1`.
   Either make it truly full or rename it so it stops spreading a false claim.

6. Keep the three docs synchronized.
   Any change to defaults, output layout, or reproduction steps should update `README.md`, `PROJECT_MAP.md`, and `idea.md` in the same change.

7. Choose one canonical stage1 root and make the code match it.
   Right now the CLI default is `outputs/stage1`, the stock stage2 config still expects `outputs/stage1_sharedbeta`, and best-of can create `outputs/stage1_bestof`. That is a taxonomy problem, not flexibility.

8. Decide whether face semantics are optional or expected.
   Right now the code can survive without them in some paths, but semantic-policy paths clearly expect them.

9. Keep reviewable source separate from vendored runtime bulk.
   If every semantic update drags a mountain of runtime files into the same diff, maintenance quality will keep dropping.
