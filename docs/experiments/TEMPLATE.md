# Experiment: Short Name

- **Date:** YYYY-MM-DD
- **Status:** planned | running | completed | stopped
- **Owner:**
- **Source commit:**
- **Run artifacts:** `runs/<experiment>/run_<timestamp>/`

## Question

State the hypothesis or decision this experiment is intended to resolve.

## Method

Link the relevant architecture, dataset, training, matching, and evaluation
documents. Describe only deviations from those shared policies.

## Configuration and provenance

| Item | Value |
| --- | --- |
| Base model and checksum | |
| Starting checkpoint and checksum | |
| Dataset and split | |
| Manifest checksum | |
| Pose schema version | |
| Seed | |
| Device and dtype | |
| Epoch range | |

Record the full command or point to the archived `run_config.json`. Include
important ablation parameters explicitly so the question remains visible.

## Results

| Checkpoint | Selection role | Primary metric | Coverage | Guardrail |
| --- | --- | ---: | ---: | ---: |
| Starting checkpoint | baseline | | | |
| Selected checkpoint | best | | | |
| Final checkpoint | final | | | |

State metric units, split, calibration state, and IoU threshold in the table or
its introduction. Do not report a conditional pose metric without its match
coverage and end-to-end counterpart.

## Observations

- What changed relative to the baseline?
- Did localization, conditional pose quality, or both change?
- Were there failures, restarts, or deviations from the intended method?

## Conclusion

State the decision supported by the run and the next experiment, if any.

## Artifacts

List the local or durable artifact locations. Note any artifacts that are not
recoverable, and avoid committing large generated files.
