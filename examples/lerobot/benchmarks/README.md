# LeRobot and FastWAM Benchmarks

This directory contains the scripts used to convert the local `flatten_box`
LeRobot dataset into RoboDM and benchmark loading speed against LeRobot.

## Scripts

- `flatten_box_robodm_benchmark.py` converts LeRobot episodes into `.vla`
  trajectories and benchmarks sequential full-frame materialization.
- `fastwam_window_benchmark.py` benchmarks the FastWAM-style training access
  pattern: 33 consecutive frames per camera plus 32 actions per sample, followed
  by FastWAM's video subsampling to 9 frames.
- `flatten_box_benchmark_summary.json` records the completed local run results.

## Local Run

The local benchmark used:

```bash
HF_DATASETS_CACHE=/private/tmp/hf_datasets \
HF_HOME=/private/tmp/hf_home \
PYTHONPATH=/Users/pfb30/lute/robodm \
/Users/pfb30/lute/quality_processor/.venv/bin/python \
  examples/lerobot/benchmarks/flatten_box_robodm_benchmark.py
```

FastWAM-shaped timing:

```bash
HF_DATASETS_CACHE=/private/tmp/hf_datasets \
HF_HOME=/private/tmp/hf_home \
PYTHONPATH=/Users/pfb30/lute/robodm \
/Users/pfb30/lute/quality_processor/.venv/bin/python \
  examples/lerobot/benchmarks/fastwam_window_benchmark.py \
  --windows 200 \
  --mode episode0
```

Generated `.vla` payloads are intentionally not committed.
