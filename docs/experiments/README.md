# Experiment Records

This directory contains concise, versioned records of training and evaluation
runs. Keep reusable architecture, data, matching, and evaluation policies in
the parent `docs/` directory; link to them from an experiment record instead of
copying them.

Local outputs live under [`runs/`](../../runs/README.md) and are intentionally
not versioned. An experiment record preserves the question, configuration,
provenance identifiers, key metrics, conclusions, and artifact location after
large checkpoints and logs move elsewhere.

## Index

| Date | Experiment | Status | Record |
| --- | --- | --- | --- |
| 2026-08-01 | ABC pose joint-lite continuation | completed | [record](2026-08-01-abc-pose-joint-lite.md) |

## Adding an experiment

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `YYYY-MM-DD-short-name.md`.
2. Link the reusable method documents and state only the experiment's
   deviations or choices.
3. Record the source commit before launch and checksums for the manifest,
   model, and starting checkpoint when available.
4. Add the record to the index above.
5. Keep checkpoints, full CSV logs, plots, and debug images under `runs/` or in
   durable artifact storage, not in Git.

Use one record for one experimental question. Multiple restarts with the same
question can be listed as separate invocations within that record.
