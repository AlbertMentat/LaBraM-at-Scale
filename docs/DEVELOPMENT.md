# Development checks

Run the CPU regression suite from the repository root:

```bash
python -B -m unittest discover -s tests -v
```

The tests cover downstream model selection and resume, saved linear probes, Quark launcher paths, multitask cache identity and checkpoint recovery. Cache tests use synthetic EEG and a small CPU encoder to exercise extraction, shard merging and training loaders. Launcher tests record the arguments passed to `torchrun` without starting training.

## Multitask checkpoints

FSDP checkpoints gather projector and LoRA optimizer state onto rank 0 on CPU. On resume, each rank reconstructs its optimizer shards. Checkpoints also store each rank's PyTorch RNG state and the task and prompt sampler states.

To compare uninterrupted and resumed training on two CUDA GPUs:

```bash
EEGLLM_TEST_DISTRIBUTED=1 EEGLLM_TEST_DEVICE=cuda \
python -B -m unittest discover -s tests -p test_multitask_checkpoint.py -v
```

This test checks replicated projector parameters, sharded LoRA parameters, AdamW moments and dropout RNG. It is skipped unless distributed testing is explicitly enabled.
