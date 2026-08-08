# Precision auditing research log

## Problem

Choose the lowest tested model dtype that satisfies an explicit numerical error
budget without blindly recommending FP32 or FP64 everywhere.

## Existing mechanisms

PyTorch can copy a module to a dtype and exposes module hooks. Autocast selects
policies but does not establish application-specific error acceptability.

## Experiment and decision

The audit deep-copies a model per candidate dtype, runs the same recursively cast
inputs, captures bounded module outputs, and compares them with a deep-copied
FP64 model. Candidate order is user-controlled. The first dtype passing absolute
and relative budgets becomes the recommendation.

## Limitations

Module output error includes upstream error, so this is not isolated per-kernel
analysis. Some devices do not implement every dtype. Stochastic state must be
controlled by the caller. Capture is bounded by element count. The audit never
rewrites the original model; automatic optimization remains intentionally absent.

