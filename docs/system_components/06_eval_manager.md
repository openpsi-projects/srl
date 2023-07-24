# System Components: Evaluation Manager

_"Without proper self-evaluation, failure is inevitable."_

## Initialization (Configuration)
Refer to the initialization docstring of [EvalManager configuration](../user_guide/config_your_experiment.md#EvalManager) on how to config an Evaluation Manager.

## Worker Logic

Evaluation manager accepts samples of two different kinds:

1. Samples of the same version to the current `eval_tag`.
2. Samples of tagged policy versions.

NOTE: eval_manager will discard samples where the policy version is not unique, or it does not match the above.

On receiving 1, the sample is considered as an evaluation result. 
With the specified frequency, data will be logged to W&B.

On receiving 2, eval_manager will extract the `episode_info` from the last step and update the metadata on that version 
accordingly.

# Related Files and Directory
- [system/impl/eval_manager.py](../../src/rlsrl/system/impl/eval_manager.py)

# What's next
- [Inference Stream](07_inference_stream.md)