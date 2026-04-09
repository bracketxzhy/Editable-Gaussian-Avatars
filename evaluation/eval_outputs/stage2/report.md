# Stage2 Supervision Evaluation Report

- Frame count: 69
- Target layer: inner
- Stage2 checkpoint: `/data/work/Editable Gaussian Avatars/outputs/stage2/gaussians3d.pt`
- Worst overall part: face

## Global Summary

- Mean MAE: 0.064291
- Mean PSNR: 16.5736
- Mean SSIM: 0.873729
- Mean Mask IoU: 0.889901

## Worst Parts

- `face` | group=exposed | badness_mean=0.255081 | mae_mean=0.092290 | ssim_mean=0.837226 | coverage_mean=0.999983
- `jewelry` | group=covered | badness_mean=0.234508 | mae_mean=0.147146 | ssim_mean=0.912638 | coverage_mean=1.000000
- `hands` | group=exposed | badness_mean=0.230454 | mae_mean=0.124042 | ssim_mean=0.894472 | coverage_mean=0.999115
- `top` | group=covered | badness_mean=0.196852 | mae_mean=0.044009 | ssim_mean=0.847157 | coverage_mean=1.000000
- `torso` | group=exposed | badness_mean=0.194865 | mae_mean=0.061104 | ssim_mean=0.866238 | coverage_mean=1.000000
- `hair` | group=covered | badness_mean=0.186421 | mae_mean=0.052601 | ssim_mean=0.866185 | coverage_mean=0.999995
- `arms` | group=exposed | badness_mean=0.170098 | mae_mean=0.086125 | ssim_mean=0.916027 | coverage_mean=1.000000
- `pants` | group=covered | badness_mean=0.164949 | mae_mean=0.044312 | ssim_mean=0.879363 | coverage_mean=1.000000
- `skirt` | group=covered | badness_mean=0.104929 | mae_mean=0.050045 | ssim_mean=0.945116 | coverage_mean=1.000000
- `feet` | group=exposed | badness_mean=0.103388 | mae_mean=0.043563 | ssim_mean=0.940175 | coverage_mean=1.000000

## Worst Frames

- `000043.jpg` | badness=0.363586 | mae=0.101025 | ssim=0.737440 | worst_parts=hands, arms, top
- `000055.jpg` | badness=0.336740 | mae=0.090285 | ssim=0.753544 | worst_parts=hands, arms, top
- `000017.jpg` | badness=0.325238 | mae=0.082885 | ssim=0.757647 | worst_parts=hands, arms, top
- `000067.jpg` | badness=0.319827 | mae=0.102010 | ssim=0.782183 | worst_parts=arms, hands, face
- `000032.jpg` | badness=0.308834 | mae=0.081957 | ssim=0.773123 | worst_parts=hands, arms, torso
- `000007.jpg` | badness=0.291585 | mae=0.069992 | ssim=0.778407 | worst_parts=hands, arms, torso
- `000046.jpg` | badness=0.265784 | mae=0.104444 | ssim=0.838746 | worst_parts=face, pants, torso
- `000031.jpg` | badness=0.262836 | mae=0.061930 | ssim=0.799095 | worst_parts=hands, torso, arms
- `000020.jpg` | badness=0.260430 | mae=0.097921 | ssim=0.837589 | worst_parts=face, pants, torso
- `000019.jpg` | badness=0.257298 | mae=0.070669 | ssim=0.813395 | worst_parts=hands, top, face
