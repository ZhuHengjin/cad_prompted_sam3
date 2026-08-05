
## Pose Head Architecture

```
Detection token (256)
Box center + extent (4)
CAD aspect-ratio features (3)
             │
             ▼
Shared trunk: 263 → 256 → 256
Each layer: Linear → LayerNorm → GELU
             │
     ┌───────┼──────────────┬─────────────┐
     │       │              │             │
   Center  Rotation       Quality     Depth fusion
   branch   branch         branch          │
     │       │              │        + metric dimensions (3)
     │       │              │        + camera features (8)
     │       │              │              │
     │       │              │         267 → 256
     │       │              │     Linear → LN → GELU
     │       │              │              │
     ▼       ▼              ▼              ▼
  2 values  6D rotation   1 logit      Depth branch
                                       1 log-depth
```

### Updated Architecture (v4: CAD Prompting and Deep Supervision)

```text
                         Encoded CAD exemplar tokens (K x 256)
                         + exemplar padding mask
                                      │
                                 LayerNorm
                                      │ keys / values
                                      ▼
Detector layers 3, 4, 5 ──┐     8-head cross-attention
  (training only)         ├──► query = LayerNorm(detection token)
Detector layer 6 ─────────┘                 │
  (final + inference)                       ▼
                              detection token + g · attention
                                   g is learned (initially 0.1)
                                              │
                                      pose token (256)
                                              │
                         ┌────────────────────┼────────────────────┐
                         │                    │                    │
          box center + extent (4)       scale-free CAD shape (3)   │
                         │                    │                    │
                         └────────────────────┴────────────────────┘
                                              │
                                              ▼
                                Shared trunk: 263 → 256 → 256
                         Each layer: Linear → LayerNorm → GELU
                                              │
                  ┌───────────────────┬───────┴────────┬──────────────────┐
                  │                   │                │                  │
                  ▼                   ▼                ▼                  ▼
             Center branch      Rotation branch  Quality branch      Depth fusion
               2 values          6D rotation       1 logit                │
                                                                    + normalized
                                                              metric dimensions (3)
                                                              + camera features (8)
                                                                          │
                                                                      267 → 256
                                                                   Linear → LN → GELU
                                                                          │
                                                                          ▼
                                                                    Depth branch
                                                                    1 log-depth

The original layer-6 detector token also continues unchanged to the segmentation path.
The same pose head is reused at every supervised detector layer: layers 3–5 use
center + depth + rotation losses, while layer 6 also uses the quality loss and
is the only layer evaluated at inference.
```

## Important Training and Model Decisions

### Current training loss (deep-prompt joint-lite)

For each image, the current run minimizes

$$
L_{total}=L_{anchor}+L_{pose}^{final}
+0.5\,\operatorname{mean}_{l\in\{3,4,5\}}L_{pose}^{l}.
$$

**Detection anchors** keep the detector and masks stable:

$$
L_{anchor}=0.10L_{mask}+0.25L_{box}+0.25L_{objectness},
\qquad L_{mask}=2L_{BCE}+2L_{Dice}.
$$

Here, $L_{mask}$ is the matched mask loss, $L_{box}$ is L1 loss on normalized
boxes, and $L_{objectness}$ is BCE on detection scores (internal positive and
negative weights `0.3` and `0.45`). These anchors use one-to-many mask matching.

**Final-layer pose loss** uses detector layer 6:

$$
L_{pose}^{final}=L_{center}+L_{depth}+L_{rotation}+L_{quality}.
$$

| Term | What it supervises |
| --- | --- |
| $L_{center}$ | Smooth-L1 on the normalized 2D projected object center $(u,v)$. |
| $L_{depth}$ | Smooth-L1 on normalized log depth. |
| $L_{rotation}$ | Smooth-L1 on one-sided nearest-neighbor CAD point-set distances after rotation, normalized by the object's 3D diagonal (`beta=0.01`). |
| $L_{quality}$ | BCE on a detached soft pose-quality target combining 3D centroid error and full placed-point-set error. |

All four pose weights are currently `1.0`. The optional trainable full-placement
term has weight `0`; full placement is used only to construct the quality
target. Translation is reconstructed from predicted center and depth, so there
is no separate direct translation-loss term.

**Auxiliary pose loss** applies the same shared pose head to detector layers
3, 4, and 5. It averages only the center, depth, and rotation terms across
those layers; quality supervision is final-layer only. The averaged auxiliary
loss has total weight `0.5`.

**Matching gate:** pose supervision uses one-to-one mask matches and is applied
only when the final predicted mask has IoU $\ge 0.5$ with its ground truth.
About 75% of matches currently pass this gate. Low-IoU samples still receive
the detection-anchor losses above; the same accepted final-layer matches are
reused for auxiliary pose supervision.
