# data/

This directory is for datasets and dataset conversion utilities.

For Milestone 1 there are **no real datasets** wired in. Training uses a toy
random matrix batch generated in `training/toy_batch.py`. See
[`dataset_plan.md`](dataset_plan.md) for the Stage 2 dataset plan.

On Rosie, the shared `/data` share is also available and mounted at the same path
on every compute node; large datasets should live there rather than in this repo.
