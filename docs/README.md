# Project Docs

This folder holds project-specific notes for CAD-prompted SAM3 work. Upstream MuggledSAM documentation remains in the package and example folders where it came from.

## Fine-Tuning

- [Fine-tuning notes](fine-tune-note.md): current LEGO SAM3 continuation command, paths, logging, plotting, and checkpoint-selection workflow.
- [Fine-tuning split behavior](finetune-split-behavior.md): frame-level train/validation/test split behavior for `finetune_image_exemplar_multi_gt_split.py`.
- [Current training loss](current-training-loss.md): the multi-GT mask, box, and presence objective used by `finetune_image_exemplar_multi_gt.py`, with source-line links.

## Model Extensions

- [CAD pose-head plan](cad-pose-head-plan.md): architecture, representation, symmetry-aware supervision, data contract, training stages, evaluation, and implementation roadmap for a YOPO-inspired 9-DoF head.
- [Perseve pose-label and dataset format](perseve-pose-dataset-format.md): canonical frames, camera-space pose capture, versioned sidecars, scale modes, mask identity, validation, and optional BOP export for Perseve synthetic data.
