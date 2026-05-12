# Data Tools

This directory contains dataset conversion and data-preparation scripts.

- `convert_post_data_01_to_dexdata.py`: base `post_data_01` converter
- `convert_post_data_01_to_dexdata_stateful.py`: stateful variant that keeps `delta_tcp` in state

These tools are kept separate from `openloop/` so evaluation code and data-prep code do not mix.

