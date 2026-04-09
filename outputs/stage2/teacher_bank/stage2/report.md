# Part Teacher Bank

Teacher selection from FASHN-2D vs SMPL-projected agreement.

## face
candidate_count: 50
selection_mode: single_teacher_best_match
- 000050.jpg | score=0.8986 | iou=0.8772 | precision=0.9212 | recall=0.9484 | centroid_px=0.49

## hair
candidate_count: 69
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 1, 'right': 2, 'back': 2, 'left': 2, 'top': 2}
bucket_max_projected_pixels: {'front': 2018, 'right': 3136, 'back': 3697, 'left': 3349, 'top': 3827}
bucket_teacher_counts: {'front': 1, 'right': 2, 'back': 2, 'left': 1, 'top': 2}
- 000053.jpg | score=0.6583 | iou=0.5937 | precision=0.9841 | recall=0.5995 | centroid_px=4.38
- 000015.jpg | score=0.7363 | iou=0.6893 | precision=0.9834 | recall=0.6974 | centroid_px=4.03
- 000043.jpg | score=0.7310 | iou=0.6799 | precision=0.9807 | recall=0.6891 | centroid_px=3.26
- 000055.jpg | score=0.7587 | iou=0.7231 | precision=0.9432 | recall=0.7561 | centroid_px=4.52
- 000033.jpg | score=0.7494 | iou=0.7078 | precision=0.9860 | recall=0.7150 | centroid_px=4.64
- 000056.jpg | score=0.7399 | iou=0.6985 | precision=0.9982 | recall=0.6994 | centroid_px=5.61
- 000032.jpg | score=0.7774 | iou=0.7358 | precision=0.9997 | recall=0.7360 | centroid_px=3.35
- 000007.jpg | score=0.7661 | iou=0.7266 | precision=0.9811 | recall=0.7369 | centroid_px=4.03

## top
candidate_count: 69
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 2, 'right': 1, 'back': 2, 'left': 2}
bucket_max_projected_pixels: {'front': 31504, 'right': 26835, 'back': 31091, 'left': 27639}
bucket_teacher_counts: {'front': 3, 'right': 1, 'back': 10, 'left': 2}
- 000023.jpg | score=0.9291 | iou=0.9166 | precision=0.9616 | recall=0.9514 | centroid_px=1.03
- 000071.jpg | score=0.9186 | iou=0.9038 | precision=0.9426 | recall=0.9565 | centroid_px=1.06
- 000026.jpg | score=0.9164 | iou=0.9057 | precision=0.9234 | recall=0.9793 | centroid_px=2.29
- 000016.jpg | score=0.8813 | iou=0.8793 | precision=0.9452 | recall=0.9265 | centroid_px=6.18
- 000013.jpg | score=0.9494 | iou=0.9434 | precision=0.9834 | recall=0.9587 | centroid_px=1.51
- 000059.jpg | score=0.9462 | iou=0.9337 | precision=0.9835 | recall=0.9485 | centroid_px=0.12
- 000011.jpg | score=0.9461 | iou=0.9365 | precision=0.9868 | recall=0.9484 | centroid_px=0.85
- 000039.jpg | score=0.9424 | iou=0.9358 | precision=0.9832 | recall=0.9509 | centroid_px=1.76
- 000035.jpg | score=0.9421 | iou=0.9386 | precision=0.9756 | recall=0.9611 | centroid_px=2.50
- 000036.jpg | score=0.9396 | iou=0.9327 | precision=0.9842 | recall=0.9469 | centroid_px=1.87
- 000014.jpg | score=0.9386 | iou=0.9301 | precision=0.9817 | recall=0.9465 | centroid_px=1.54
- 000063.jpg | score=0.9380 | iou=0.9308 | precision=0.9846 | recall=0.9446 | centroid_px=1.88
- 000038.jpg | score=0.9377 | iou=0.9326 | precision=0.9746 | recall=0.9558 | centroid_px=2.36
- 000062.jpg | score=0.9376 | iou=0.9275 | precision=0.9821 | recall=0.9435 | centroid_px=1.22
- 000034.jpg | score=0.9308 | iou=0.9224 | precision=0.9681 | recall=0.9513 | centroid_px=1.95
- 000009.jpg | score=0.9257 | iou=0.9189 | precision=0.9656 | recall=0.9500 | centroid_px=2.61

## dress
candidate_count: 0
selection_mode: three_d_bucket_two_d_consistency
no usable teacher

## skirt
candidate_count: 0
selection_mode: three_d_bucket_two_d_consistency
no usable teacher

## pants
candidate_count: 69
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 2, 'right': 1, 'back': 2, 'left': 2}
bucket_max_projected_pixels: {'front': 20439, 'right': 16859, 'back': 20126, 'left': 18759}
bucket_teacher_counts: {'front': 2, 'right': 1, 'back': 2, 'left': 2}
- 000003.jpg | score=0.8920 | iou=0.8826 | precision=0.9727 | recall=0.9050 | centroid_px=3.92
- 000072.jpg | score=0.8813 | iou=0.8829 | precision=0.9641 | recall=0.9129 | centroid_px=7.16
- 000044.jpg | score=0.8607 | iou=0.8482 | precision=0.9764 | recall=0.8659 | centroid_px=5.02
- 000013.jpg | score=0.8591 | iou=0.8612 | precision=0.9830 | recall=0.8742 | centroid_px=8.74
- 000039.jpg | score=0.8560 | iou=0.8560 | precision=0.9879 | recall=0.8651 | centroid_px=8.46
- 000035.jpg | score=0.8773 | iou=0.8497 | precision=0.9938 | recall=0.8542 | centroid_px=0.64
- 000057.jpg | score=0.8627 | iou=0.8457 | precision=0.9803 | recall=0.8603 | centroid_px=3.85

## belt
candidate_count: 0
selection_mode: cluster_topk
no usable teacher

## bag
candidate_count: 0
selection_mode: cluster_topk
no usable teacher

## hat
candidate_count: 0
selection_mode: cluster_topk
no usable teacher

## scarf
candidate_count: 0
selection_mode: cluster_topk
no usable teacher

## glasses
candidate_count: 0
selection_mode: cluster_topk
no usable teacher

## arms
candidate_count: 59
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 2, 'right': 1, 'back': 2, 'left': 2}
bucket_max_projected_pixels: {'front': 3389, 'right': 3026, 'back': 3048, 'left': 3420}
bucket_teacher_counts: {'front': 2, 'right': 1, 'back': 2, 'left': 2}
- 000051.jpg | score=0.8176 | iou=0.7827 | precision=0.9606 | recall=0.8086 | centroid_px=1.77
- 000074.jpg | score=0.7890 | iou=0.7552 | precision=0.8994 | recall=0.8248 | centroid_px=2.70
- 000063.jpg | score=0.8135 | iou=0.7747 | precision=0.9611 | recall=0.7998 | centroid_px=1.07
- 000012.jpg | score=0.8086 | iou=0.7784 | precision=0.9793 | recall=0.7915 | centroid_px=3.79
- 000038.jpg | score=0.7835 | iou=0.7481 | precision=0.9709 | recall=0.7653 | centroid_px=3.78
- 000035.jpg | score=0.8193 | iou=0.8085 | precision=0.9602 | recall=0.8365 | centroid_px=7.53
- 000072.jpg | score=0.8183 | iou=0.7847 | precision=0.9651 | recall=0.8077 | centroid_px=2.14

## hands
candidate_count: 54
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 1, 'right': 2, 'left': 2}
bucket_max_projected_pixels: {'front': 1942, 'right': 2037, 'left': 2320}
bucket_teacher_counts: {'front': 1, 'right': 2, 'left': 2}
- 000027.jpg | score=0.6949 | iou=0.6572 | precision=0.9192 | recall=0.6974 | centroid_px=6.83
- 000074.jpg | score=0.7067 | iou=0.6541 | precision=0.9742 | recall=0.6656 | centroid_px=4.10
- 000038.jpg | score=0.6936 | iou=0.6442 | precision=0.9637 | recall=0.6602 | centroid_px=5.31
- 000057.jpg | score=0.7067 | iou=0.6697 | precision=0.9483 | recall=0.6951 | centroid_px=7.14
- 000012.jpg | score=0.6866 | iou=0.6395 | precision=0.9626 | recall=0.6558 | centroid_px=6.24

## legs
candidate_count: 0
selection_mode: sequence_buckets
no usable teacher

## feet
candidate_count: 68
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'front': 1, 'right': 2, 'back': 2, 'left': 2, 'top': 2}
bucket_max_projected_pixels: {'front': 2339, 'right': 2804, 'back': 2567, 'left': 2729, 'top': 2488}
bucket_teacher_counts: {'front': 1, 'right': 2, 'back': 2, 'left': 2, 'top': 2}
- 000000.jpg | score=0.7774 | iou=0.7341 | precision=0.9446 | recall=0.7671 | centroid_px=1.61
- 000047.jpg | score=0.8216 | iou=0.7923 | precision=0.9344 | recall=0.8389 | centroid_px=2.54
- 000020.jpg | score=0.8201 | iou=0.7865 | precision=0.9829 | recall=0.7974 | centroid_px=2.35
- 000035.jpg | score=0.7946 | iou=0.7603 | precision=0.9643 | recall=0.7824 | centroid_px=3.27
- 000033.jpg | score=0.7895 | iou=0.7474 | precision=0.9420 | recall=0.7835 | centroid_px=1.22
- 000034.jpg | score=0.7994 | iou=0.7625 | precision=0.9029 | recall=0.8306 | centroid_px=1.43
- 000009.jpg | score=0.7855 | iou=0.7456 | precision=0.9753 | recall=0.7600 | centroid_px=2.69
- 000028.jpg | score=0.7683 | iou=0.7244 | precision=0.9719 | recall=0.7399 | centroid_px=2.61
- 000069.jpg | score=0.6985 | iou=0.6256 | precision=0.9985 | recall=0.6261 | centroid_px=0.52

## torso
candidate_count: 69
selection_mode: three_d_bucket_two_d_consistency
bucket_minimums: {'right': 1, 'back': 2, 'left': 2, 'bottom': 2}
bucket_max_projected_pixels: {'right': 1861, 'back': 2110, 'left': 1991, 'bottom': 2421}
bucket_teacher_counts: {'right': 1, 'back': 2, 'left': 2, 'bottom': 2}
- 000043.jpg | score=0.8334 | iou=0.8083 | precision=0.9035 | recall=0.8846 | centroid_px=2.75
- 000010.jpg | score=0.9047 | iou=0.8854 | precision=0.9439 | recall=0.9346 | centroid_px=0.67
- 000059.jpg | score=0.8781 | iou=0.8585 | precision=0.9227 | recall=0.9249 | centroid_px=1.96
- 000057.jpg | score=0.8709 | iou=0.8454 | precision=0.8886 | recall=0.9457 | centroid_px=0.97
- 000003.jpg | score=0.8368 | iou=0.8048 | precision=0.9437 | recall=0.8454 | centroid_px=1.18
- 000048.jpg | score=0.8572 | iou=0.8367 | precision=0.9143 | recall=0.9080 | centroid_px=2.78
- 000022.jpg | score=0.8354 | iou=0.8093 | precision=0.9211 | recall=0.8696 | centroid_px=2.50

## jewelry
candidate_count: 0
selection_mode: cluster_topk
no usable teacher
