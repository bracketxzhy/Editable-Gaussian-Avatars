from __future__ import annotations
import subprocess
import sys


def run(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    py = sys.executable

    run([py, "-m", "avatar_gs.cli.preprocess_video", "--config", "configs/video.yaml"])
    run([
        py,
        "-m",
        "avatar_gs.cli.stage1_fit_wham_pymaf",
        "--project_root",
        ".",
        "--frames_dir",
        "data/frames/input",
        "--assets_dir",
        "assets",
        "--out_dir",
        "outputs/stage1",
        "--pymaf_frame_indices",
        "-1",
        "--single_person",
    ])
    run([py, "-m", "avatar_gs.cli.stage2_train_gaussian", "--config", "configs/stage2_gaussian.yaml"])
    run([py, "-m", "avatar_gs.cli.render", "--config", "configs/render.yaml"])


if __name__ == "__main__":
    main()
