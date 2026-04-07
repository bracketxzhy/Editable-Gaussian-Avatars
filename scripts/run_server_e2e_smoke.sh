#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/ega/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-server_e2e_${STAMP}}"
RUN_ROOT_REL="outputs/${RUN_NAME}"
RUN_ROOT="${ROOT}/${RUN_ROOT_REL}"
FRAMES_REL="${RUN_ROOT_REL}/frames"
STAGE1_REL="${RUN_ROOT_REL}/stage1"
STAGE2_REL="${RUN_ROOT_REL}/stage2"
RENDER_REL="${RUN_ROOT_REL}/render"
LOG_REL="${RUN_ROOT_REL}/logs"
SUMMARY_PATH="${RUN_ROOT}/summary.txt"
MAX_FRAMES="${MAX_FRAMES:-8}"
STAGE2_N_FRAMES="${STAGE2_N_FRAMES:-2}"
STAGE2_ITERS="${STAGE2_ITERS:-1}"
RENDER_MAX_FRAMES="${RENDER_MAX_FRAMES:-1}"
SKIP_FASHN="${SKIP_FASHN:-0}"

mkdir -p "${RUN_ROOT}" "${ROOT}/${LOG_REL}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_stage() {
  local name="$1"
  shift
  local stdout_log="${ROOT}/${LOG_REL}/${name}.stdout.log"
  local stderr_log="${ROOT}/${LOG_REL}/${name}.stderr.log"
  echo "[$(date '+%F %T')] RUN ${name}: $*"
  "$@" > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)
}

cd "${ROOT}"
run_stage ensure_assets "${PYTHON_BIN}" scripts/ensure_smplx_flame_vertex_ids.py

run_stage preprocess   "${PYTHON_BIN}" -m avatar_gs.cli.preprocess_video   --config configs/video.yaml   --override   project_root=.   log_dir="${LOG_REL}"   video.out_frames_dir="${FRAMES_REL}"   video.max_frames="${MAX_FRAMES}"

stage1_args=(
  "${PYTHON_BIN}" -m avatar_gs.cli.stage1_fit_smplestx
  --project_root .
  --frames_dir "${FRAMES_REL}"
  --assets_dir assets
  --out_dir "${STAGE1_REL}"
  --smplestx_frame_indices -1
  --single_person
)
if [[ "${SKIP_FASHN}" == "1" ]]; then
  stage1_args+=(--skip_fashn_face_semantics)
fi
run_stage stage1 "${stage1_args[@]}"

run_stage stage2   "${PYTHON_BIN}" -m avatar_gs.cli.stage2_train_gaussian   --config configs/stage2_gaussian.yaml   --override   project_root=.   log_dir="${LOG_REL}"   data.frames_dir="${FRAMES_REL}"   stage2.stage1_dir="${STAGE1_REL}"   stage2.out_dir="${STAGE2_REL}"   stage2.inner_ckpt_path="${STAGE2_REL}/gaussians3d_inner.pt"   stage2.prepared_supervision_cache_path="${STAGE2_REL}/supervision_cache.npz"   stage2.keypoints_json="${STAGE1_REL}/keypoints.json"   stage2.bind_mesh_sequence_dir="${STAGE1_REL}"   stage2.bind_sequence_npz="${STAGE1_REL}/smpl_params_sequence.npz"   stage2.person_masks_npz="${STAGE1_REL}/person_masks.npz"   stage2.smpl_face_semantics_npz="${STAGE1_REL}/fashn_face_semantics/smpl_face_semantics.npz"   stage2.n_frames="${STAGE2_N_FRAMES}"   stage2.iters="${STAGE2_ITERS}"

run_stage render   "${PYTHON_BIN}" -m avatar_gs.cli.render   --config configs/render.yaml   --override   project_root=.   log_dir="${LOG_REL}"   render.frames_dir="${FRAMES_REL}"   render.stage1_dir="${STAGE1_REL}"   render.stage2_dir="${STAGE2_REL}"   render.prepared_supervision_cache_path="${STAGE2_REL}/supervision_cache.npz"   render.out_dir="${RENDER_REL}"   render.max_frames="${RENDER_MAX_FRAMES}"   render.person_masks_npz="${STAGE1_REL}/person_masks.npz"   render.smpl_skeleton_npz="${STAGE1_REL}/smpl_skeleton_sequence.npz"   render.smpl_joints_npz="${STAGE1_REL}/smpl_joints_sequence.npz"   render.bind_mesh_sequence_dir="${STAGE1_REL}"   render.bind_sequence_npz="${STAGE1_REL}/smpl_params_sequence.npz"

{
  echo "run_name=${RUN_NAME}"
  echo "run_root=${RUN_ROOT}"
  echo "frames_dir=${ROOT}/${FRAMES_REL}"
  echo "stage1_dir=${ROOT}/${STAGE1_REL}"
  echo "stage2_dir=${ROOT}/${STAGE2_REL}"
  echo "render_dir=${ROOT}/${RENDER_REL}"
  echo "summary_generated_at=$(date '+%F %T')"
  find "${ROOT}/${STAGE1_REL}" -maxdepth 2 -type f | sort
  find "${ROOT}/${STAGE2_REL}" -maxdepth 2 -type f | sort
  find "${ROOT}/${RENDER_REL}" -maxdepth 3 -type f | sort
} > "${SUMMARY_PATH}"

echo "Summary written to ${SUMMARY_PATH}"
