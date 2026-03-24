param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$WaitIntervalSec = 30,
    [int]$MaxRounds = 2
)

$ErrorActionPreference = "Stop"

function Get-AvatarGsRuns {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "avatar_gs\.cli\.(stage1_fit_smplestx|stage2_train_gaussian|render)"
        }
}

Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs/logs"
$null = New-Item -ItemType Directory -Path $logDir -Force
$logFile = Join-Path $logDir "auto_full_pipeline.log"

function Log([string]$Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    $line | Tee-Object -FilePath $logFile -Append | Out-Null
}

if ($MaxRounds -lt 1) {
    $MaxRounds = 1
}

Log "Auto pipeline watcher started. RepoRoot=$RepoRoot"
Log "Waiting for existing avatar_gs tasks to finish..."

while ($true) {
    $runs = Get-AvatarGsRuns
    if ($runs.Count -eq 0) {
        break
    }
    $pids = ($runs | Select-Object -ExpandProperty ProcessId) -join ","
    Log "Detected running task(s), PID=$pids. Sleep $WaitIntervalSec sec."
    Start-Sleep -Seconds $WaitIntervalSec
}

Log "No running avatar_gs task detected. Starting full pipeline."

Log "Stage1 start"
python -m avatar_gs.cli.stage1_fit_smplestx --project_root . --frames_dir data/frames/input --assets_dir assets --out_dir outputs/stage1 --smplestx_frame_indices -1 --single_person 2>&1 |
    Tee-Object -FilePath $logFile -Append | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Stage1 failed with exit code $LASTEXITCODE"
}
Log "Stage1 done"

$roundOverrides = @(
    @(),
    @(
        "stage2.lr_gauss=0.0012",
        "stage2.fg_opacity_logit_max=1.5",
        "stage2.alpha_threshold=0.08"
    )
)

if ($MaxRounds -gt $roundOverrides.Count) {
    $MaxRounds = $roundOverrides.Count
}

$bestPsnr = -1.0
$bestScore = -1.0e9
$bestRound = -1
$bestCkptBytes = $null
$bestRenderBytes = $null

for ($i = 0; $i -lt $MaxRounds; $i++) {
    $round = $i + 1
    $ov = $roundOverrides[$i]

    if ($ov.Count -gt 0) {
        Log "Round $round Stage2 start with overrides: $($ov -join ', ')"
        $args = @(
            "-m", "avatar_gs.cli.stage2_train_gaussian",
            "--config", "configs/stage2_gaussian.yaml",
            "--override"
        ) + $ov
    }
    else {
        Log "Round $round Stage2 start (default config)"
        $args = @(
            "-m", "avatar_gs.cli.stage2_train_gaussian",
            "--config", "configs/stage2_gaussian.yaml"
        )
    }

    & python @args 2>&1 | Tee-Object -FilePath $logFile -Append | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Stage2 round $round failed with exit code $LASTEXITCODE"
    }
    Log "Round $round Stage2 done"

    Log "Round $round render start"
    python -m avatar_gs.cli.render --config configs/stage2_gaussian.yaml 2>&1 |
        Tee-Object -FilePath $logFile -Append | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Render round $round failed with exit code $LASTEXITCODE"
    }
    Log "Round $round render done"

    $evalRaw = @'
import cv2
import math
import numpy as np
import os
import torch
import json

ckpt = torch.load(r"outputs/stage2/gaussians3d.pt", map_location="cpu")
frame = ckpt["frames"][0]
pred = cv2.imread(r"outputs/renders/fg_only/000000.png", cv2.IMREAD_COLOR)
gt = cv2.imread(os.path.join(r"data/frames/input", frame), cv2.IMREAD_COLOR)
if pred is None or gt is None:
    print(json.dumps({"ok": False, "error": "missing_pred_or_gt"}, ensure_ascii=False))
    raise SystemExit(0)
pred = pred.astype(np.float32) / 255.0
gt = gt.astype(np.float32) / 255.0

mse = float(np.mean((pred - gt) ** 2))
psnr = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
mae = float(np.mean(np.abs(pred - gt)))

diff = np.mean(np.abs(pred - gt), axis=2)
h, w = diff.shape
mask = np.ones((h, w), dtype=bool)

kp_path = r"outputs/stage1/keypoints.json"
if os.path.exists(kp_path):
    try:
        with open(kp_path, "r", encoding="utf-8") as f:
            kpts = json.load(f)
        arr = kpts.get(frame, None)
        if arr is not None:
            xs = [p[0] for p in arr if p[2] >= 0.2]
            ys = [p[1] for p in arr if p[2] >= 0.2]
            if len(xs) > 0 and len(ys) > 0:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                bw = x1 - x0
                bh = y1 - y0
                pad = 0.25
                x0 = max(0, int(round(x0 - pad * bw)))
                x1 = min(w - 1, int(round(x1 + pad * bw)))
                y0 = max(0, int(round(y0 - pad * bh)))
                y1 = min(h - 1, int(round(y1 + pad * bh)))
                mask = np.zeros((h, w), dtype=bool)
                mask[y0:y1 + 1, x0:x1 + 1] = True
    except Exception:
        pass

inside = mask
outside = ~mask

mae_in = float(diff[inside].mean()) if inside.any() else mae
mae_out = float(diff[outside].mean()) if outside.any() else 0.0
outlier_in = float((diff[inside] > 0.15).mean()) if inside.any() else 0.0
outlier_out = float((diff[outside] > 0.15).mean()) if outside.any() else 0.0

pred_u8 = (pred * 255.0).clip(0, 255).astype(np.uint8)
gt_u8 = (gt * 255.0).clip(0, 255).astype(np.uint8)
pred_gray = cv2.cvtColor(pred_u8, cv2.COLOR_BGR2GRAY)
gt_gray = cv2.cvtColor(gt_u8, cv2.COLOR_BGR2GRAY)
sharp_pred = float(cv2.Laplacian(pred_gray, cv2.CV_32F).var())
sharp_gt = float(cv2.Laplacian(gt_gray, cv2.CV_32F).var())
sharp_ratio = float(sharp_pred / (sharp_gt + 1e-6))

issues = []
if mae_out > 0.06 or outlier_out > 0.03:
    issues.append("非人体区域误差偏高，存在明显鬼影或漂浮高斯。")
if sharp_ratio < 0.45:
    issues.append("整体偏糊，边缘锐度明显低于输入帧。")
if mae_in > 0.10:
    issues.append("人体区域拟合误差较大，颜色或几何不稳定。")
if psnr < 18.0:
    issues.append("整体重建质量偏低。")
if len(issues) == 0:
    issues.append("未发现明显鬼影，当前结果可用。")

if psnr >= 24.0 and mae_out < 0.03 and sharp_ratio >= 0.70:
    grade = "A"
elif psnr >= 20.0 and mae_out < 0.05 and sharp_ratio >= 0.50:
    grade = "B"
else:
    grade = "C"

# 兼顾清晰度与鬼影的综合分，用于自动挑选最佳轮次
score = float(psnr - 35.0 * mae_out - 6.0 * max(0.0, 0.60 - sharp_ratio))

print(json.dumps({
    "ok": True,
    "frame": frame,
    "psnr": round(psnr, 4),
    "mae": round(mae, 4),
    "mae_inside": round(mae_in, 4),
    "mae_outside": round(mae_out, 4),
    "outlier_inside": round(outlier_in, 4),
    "outlier_outside": round(outlier_out, 4),
    "sharp_ratio": round(sharp_ratio, 4),
    "grade": grade,
    "score": round(score, 4),
    "issues": issues
}, ensure_ascii=False))
'@ | python -

    $eval = $evalRaw | ConvertFrom-Json
    if (-not $eval.ok) {
        throw "Eval round $round failed: $($eval.error)"
    }

    $psnr = [double]$eval.psnr
    $score = [double]$eval.score
    Log "Round $round Eval: grade=$($eval.grade), frame=$($eval.frame), score=$score, PSNR=$($eval.psnr), MAE=$($eval.mae), MAE_in=$($eval.mae_inside), MAE_out=$($eval.mae_outside), blur_ratio=$($eval.sharp_ratio), outlier_out=$($eval.outlier_outside)"
    foreach ($issue in $eval.issues) {
        Log "Round $round Issue: $issue"
    }

    if ($score -gt $bestScore) {
        $bestScore = $score
        $bestPsnr = $psnr
        $bestRound = $round
        $bestCkptBytes = [System.IO.File]::ReadAllBytes((Join-Path $RepoRoot "outputs/stage2/gaussians3d.pt"))
        $bestRenderBytes = [System.IO.File]::ReadAllBytes((Join-Path $RepoRoot "outputs/renders/fg_only/000000.png"))
        Log "Round $round becomes best."
    }
}

if ($bestRound -ge 1 -and $null -ne $bestCkptBytes -and $null -ne $bestRenderBytes) {
    [System.IO.File]::WriteAllBytes((Join-Path $RepoRoot "outputs/stage2/gaussians3d.pt"), $bestCkptBytes)
    [System.IO.File]::WriteAllBytes((Join-Path $RepoRoot "outputs/renders/fg_only/000000.png"), $bestRenderBytes)
}

Log "All rounds completed. BestRound=$bestRound BestPSNR=$bestPsnr"
Log "BestScore=$bestScore"
Log "Artifacts kept with original names: outputs/stage2/gaussians3d.pt and outputs/renders/fg_only/000000.png"


