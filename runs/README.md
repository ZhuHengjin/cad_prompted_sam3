# Run Artifacts

Keep generated training and evaluation artifacts under this directory instead of the repository root. Git ignores all contents except this guide.

Use one directory per experiment and one timestamped directory per invocation:

```text
runs/
└── finetune_exemplar_abc_pose_joint_lite/
    └── run_20260801_013714/
        ├── checkpoints/
        │   ├── finetune.pth
        │   ├── finetune_epoch_052.pth
        │   └── finetune_calibrated.pth
        ├── debug_boxes/
        ├── dataset_manifest.csv
        ├── metrics.csv
        ├── pose_provenance.json
        └── run_config.json
```

The experiment directory groups runs that share a configuration or purpose. The timestamped run directory is the unit to archive, compare, or remove. Keep checkpoints together under `checkpoints/`; keep metrics, configuration, manifest snapshots, and provenance at the run level so a run remains reproducible.

Large or important checkpoints should also be copied to durable object storage or an artifact tracker. This local `runs/` tree is working storage and is not versioned by Git.
