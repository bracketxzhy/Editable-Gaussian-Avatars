# Stage2 Supervision Evaluation Report

- Frame count: 69
- Target layer: combined
- Stage2 checkpoint: `/data/work/Editable Gaussian Avatars/outputs/stage2/eval_combined_work/combined_inner_existing_outer_d2.pt`
- Worst overall part: face

## Global Summary

- Mean MAE: 0.071335
- Mean PSNR: 15.9213
- Mean SSIM: 0.857957
- Mean Mask IoU: 0.990929

## Worst Parts

- `face` | group=exposed | badness_mean=0.434535 | mae_mean=0.135608 | ssim_mean=0.773574 | coverage_mean=0.927500
- `hands` | group=exposed | badness_mean=0.430108 | mae_mean=0.191177 | ssim_mean=0.876593 | coverage_mean=0.884476
- `feet` | group=exposed | badness_mean=0.270210 | mae_mean=0.090128 | ssim_mean=0.853422 | coverage_mean=0.966496
- `hair` | group=covered | badness_mean=0.251746 | mae_mean=0.064320 | ssim_mean=0.820374 | coverage_mean=0.992200
- `torso` | group=exposed | badness_mean=0.212916 | mae_mean=0.069515 | ssim_mean=0.857324 | coverage_mean=0.999275
- `arms` | group=exposed | badness_mean=0.205522 | mae_mean=0.101308 | ssim_mean=0.917191 | coverage_mean=0.978595
- `top` | group=covered | badness_mean=0.196011 | mae_mean=0.042627 | ssim_mean=0.847721 | coverage_mean=0.998894
- `pants` | group=covered | badness_mean=0.190373 | mae_mean=0.047650 | ssim_mean=0.858153 | coverage_mean=0.999124
- `bag` | group=covered | badness_mean=0.128698 | mae_mean=0.034602 | ssim_mean=0.905904 | coverage_mean=1.000000
- `skirt` | group=covered | badness_mean=0.056212 | mae_mean=0.033680 | ssim_mean=0.977468 | coverage_mean=1.000000

## Worst Frames

- `000043.jpg` | badness=0.391326 | mae=0.101543 | ssim=0.714677 | worst_parts=arms, hands, face
- `000017.jpg` | badness=0.357828 | mae=0.086101 | ssim=0.733631 | worst_parts=hands, feet, face
- `000019.jpg` | badness=0.345815 | mae=0.094784 | ssim=0.791081 | worst_parts=hands, feet, arms
- `000055.jpg` | badness=0.341962 | mae=0.093672 | ssim=0.755453 | worst_parts=hands, face, arms
- `000067.jpg` | badness=0.337298 | mae=0.098690 | ssim=0.763225 | worst_parts=arms, hands, face
- `000032.jpg` | badness=0.322040 | mae=0.084276 | ssim=0.766020 | worst_parts=hands, arms, face
- `000045.jpg` | badness=0.321238 | mae=0.101931 | ssim=0.827971 | worst_parts=hands, arms, face
- `000065.jpg` | badness=0.311967 | mae=0.072615 | ssim=0.781061 | worst_parts=face, arms, hands
- `000046.jpg` | badness=0.310496 | mae=0.108334 | ssim=0.817985 | worst_parts=feet, hands, face
- `000007.jpg` | badness=0.308563 | mae=0.071646 | ssim=0.766733 | worst_parts=hands, arms, face
