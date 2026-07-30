# Project Docs

This folder holds project-specific notes for CAD-prompted SAM3 work. Upstream MuggledSAM documentation remains in the package and example folders where it came from.

## Fine-Tuning

- [Fine-tuning notes](fine-tune-note.md): current LEGO SAM3 continuation command, paths, logging, plotting, and checkpoint-selection workflow.
- [Fine-tuning split behavior](finetune-split-behavior.md): frame-level train/validation/test split behavior for `finetune_image_exemplar_multi_gt_split.py`.
- [Current training loss](current-training-loss.md): the multi-GT mask, box, and presence objective used by `finetune_image_exemplar_multi_gt.py`, with source-line links.

## Model Extensions

- [CAD pose-head plan](cad-pose-head-plan.md): target architecture, AABB-frame centroid representation, label-free point-set supervision, training stages, evaluation, and migration roadmap.
- [Label-free point-set pose supervision](point-set-pose-supervision-plan.md): active geometry-artifact, centroid, loss, compute, evaluation, and acceptance contract that replaces explicit symmetry labels.
- [CAD pose-head implementation guide](cad-pose-head-implementation.md): the implemented point-set/centroid path, including architecture, dataset validation, training, inference, tests, and legacy v1 compatibility.
- [Perseve pose-label and dataset format](perseve-pose-dataset-format.md): active v2 point-set sidecar contract; legacy v1 symmetry fields remain documented for compatibility.
