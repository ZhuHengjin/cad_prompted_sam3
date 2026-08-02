# Project Docs

This folder holds project-specific notes for CAD-prompted SAM3 work. Upstream MuggledSAM documentation remains in the package and example folders where it came from.

Generated training and evaluation data follows the repository's [run artifact layout](../runs/README.md).

## Experiments

- [Experiment records](experiments/README.md): index and conventions for concise, reproducible run histories.
- [Experiment template](experiments/TEMPLATE.md): required metadata, results, observations, and artifact links for new runs.
- [ABC pose joint-lite continuation](experiments/2026-08-01-abc-pose-joint-lite.md): reconstructed configuration, validation results, limitations, and conclusion for the 2026-08-01 run.

## Fine-Tuning

- [Fine-tuning notes](fine-tune-note.md): current LEGO SAM3 continuation command, paths, logging, plotting, and checkpoint-selection workflow.
- [Fine-tuning split behavior](finetune-split-behavior.md): versioned manifest construction and train/validation/test behavior for `finetune_image_exemplar_multi_gt.py`.
- [Current training loss](current-training-loss.md): the multi-GT mask, box, and presence objective used by `finetune_image_exemplar_multi_gt.py`.

## Model Extensions

- [CAD pose-head plan](cad-pose-head-plan.md): target architecture, AABB-frame centroid representation, label-free point-set supervision, training stages, evaluation, and migration roadmap.
- [Label-free point-set pose supervision](point-set-pose-supervision-plan.md): active geometry-artifact, centroid, loss, compute, evaluation, and acceptance contract that replaces explicit symmetry labels.
- [CAD pose-head implementation guide](cad-pose-head-implementation.md): the implemented point-set/centroid path, including architecture, dataset validation, training, inference, tests, and legacy v1 compatibility.
- [CAD pose matching and evaluation](cad-pose-matching-and-evaluation.md): shared one-to-one assignment, train/eval IoU filtering, coverage metrics, calibration, and interpretation policy.
- [ABC point-set pose dataset preparation](abc-pose-v2-preparation.md): preprocess USD point sets, upgrade Perseve metadata to v2, build scene-level splits, stage and render the complete ABC STL corpus, validate, and launch a pose-head smoke run.
- [CAD pose joint-lite training](cad-pose-joint-lite-training.md): pose-first joint adaptation, anchor losses, differential learning rates, launch configuration, and checkpoint-selection criteria.
- [Perseve pose-label and dataset format](perseve-pose-dataset-format.md): active v2 point-set sidecar contract; legacy v1 symmetry fields remain documented for compatibility.
