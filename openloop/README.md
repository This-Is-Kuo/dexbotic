# Openloop Workspace

This directory centralizes the `post_data_01` open-loop workflow.

- `eval_openloop.py`: offline open-loop evaluator
- `tools/`: metrics, lag checks, resume helpers, alignment checks
- `artifacts/`: saved open-loop experiment outputs

Dataset conversion scripts live under `data_tools/`.
Root-level legacy entrypoints are kept as thin wrappers for compatibility.
