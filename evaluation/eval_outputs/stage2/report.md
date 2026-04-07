# Stage2 Supervision Evaluation Report

- Frame count: 69
- Target layer: inner
- Stage2 checkpoint: `/data/work/Editable Gaussian Avatars/outputs/stage2_part_fit_r154_partteacher_toponly_v15_greedyextra_from_r130_headboxfixed_300/gaussians3d_partfit_preview.pt`
- Worst overall part: face

## Global Summary

- Mean MAE: 0.058596
- Mean PSNR: 16.9667
- Mean SSIM: 0.881134
- Mean Mask IoU: 0.895299

## Worst Parts

- `face` | group=exposed | badness_mean=0.253219 | mae_mean=0.091675 | ssim_mean=0.838484 | coverage_mean=0.999972
- `jewelry` | group=covered | badness_mean=0.214648 | mae_mean=0.128463 | ssim_mean=0.913814 | coverage_mean=1.000000
- `hands` | group=exposed | badness_mean=0.204159 | mae_mean=0.106906 | ssim_mean=0.903307 | coverage_mean=0.999439
- `hair` | group=covered | badness_mean=0.186390 | mae_mean=0.052599 | ssim_mean=0.866221 | coverage_mean=0.999987
- `top` | group=covered | badness_mean=0.183077 | mae_mean=0.039123 | ssim_mean=0.856045 | coverage_mean=1.000000
- `torso` | group=exposed | badness_mean=0.177157 | mae_mean=0.054204 | ssim_mean=0.877047 | coverage_mean=1.000000
- `arms` | group=exposed | badness_mean=0.148064 | mae_mean=0.076816 | ssim_mean=0.928753 | coverage_mean=1.000000
- `pants` | group=covered | badness_mean=0.147996 | mae_mean=0.036302 | ssim_mean=0.888305 | coverage_mean=1.000000
- `feet` | group=exposed | badness_mean=0.091465 | mae_mean=0.037765 | ssim_mean=0.946300 | coverage_mean=1.000000
- `skirt` | group=covered | badness_mean=0.091143 | mae_mean=0.044569 | ssim_mean=0.953425 | coverage_mean=1.000000

## Worst Frames

- `000055.jpg` | badness=0.320145 | mae=0.084270 | ssim=0.764125 | worst_parts=hands, arms, top
- `000043.jpg` | badness=0.316024 | mae=0.084346 | ssim=0.768322 | worst_parts=arms, hands, feet
- `000017.jpg` | badness=0.293507 | mae=0.073591 | ssim=0.780084 | worst_parts=hands, arms, hair
- `000032.jpg` | badness=0.276449 | mae=0.071341 | ssim=0.794893 | worst_parts=hands, arms, top
- `000067.jpg` | badness=0.269990 | mae=0.083939 | ssim=0.813949 | worst_parts=arms, hands, face
- `000007.jpg` | badness=0.259371 | mae=0.061153 | ssim=0.801783 | worst_parts=hands, arms, top
- `000031.jpg` | badness=0.228855 | mae=0.052572 | ssim=0.823716 | worst_parts=hands, arms, torso
- `000006.jpg` | badness=0.225557 | mae=0.059309 | ssim=0.833751 | worst_parts=hands, arms, torso
- `000066.jpg` | badness=0.213801 | mae=0.048664 | ssim=0.834863 | worst_parts=hands, arms, face
- `000044.jpg` | badness=0.212854 | mae=0.049509 | ssim=0.836655 | worst_parts=hands, jewelry, torso
