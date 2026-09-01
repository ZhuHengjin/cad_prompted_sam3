# Quality-off + full-set pose-loss experiment

Kept color augmentation and the frozen `joint_lite` visual backbone; enabled the full-set point loss (`weight=0.1`), disabled pose-quality loss, and retained deep supervision on detector layers 3/4/5 (`aux_weight=0.5`), CAD prompting, and `view_id` conditioning.

| Dataset | Training | Pose results |
|---|---:|---|
| Overfit-50 | 500 epochs | Best train pose success **83.53%** (epoch 385); final calibrated **81.18%**; centroid error **1.83 cm**; normalized surface error **0.0549**. |
| Full 5k (4k/500/500) | 100 epochs | Best validation pose success **22.00%** (epoch 97); best IoU≥0.7 conditional **22.89%**, end-to-end **21.28%**; final calibrated **20.13%**; centroid error **4.80 cm**; normalized surface error **0.1321**. |

Conclusion: the change substantially improved memorization over the prior Overfit-50 result (**63.53% → 83.53%**), but reduced held-out 5k performance versus the prior recipe (**24.23% → 22.00%** best validation pose success).
