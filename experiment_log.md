# CS336 Assignment 1 Experiment Log

## tinystories_memory_test_20260722_152311

### Purpose

Verify that the assignment-recommended TinyStories model fits on an
NVIDIA GeForce RTX 4060 Laptop GPU, completes forward and backward
passes, and produces valid experiment logs.

### Configuration

- Dataset: TinyStories
- Vocabulary size: 10,000
- Context length: 256
- Model dimension: 512
- Feed-forward dimension: 1,344
- Transformer layers: 4
- Attention heads: 16
- Batch size: 8
- Training steps: 20
- Maximum learning rate: 1e-3
- Minimum learning rate: 1e-4
- Warmup steps: 2
- Weight decay: 0.01
- Validation interval: 10 steps
- Validation batches per evaluation: 5
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

### Results

- Tokens processed: 40,960
- Wall-clock time: 2.37 seconds
- Initial training loss: 9.2722
- Final training loss: 5.3891
- Validation loss at step 10: 5.9018
- Validation loss at step 20: 5.2639
- The checkpoint, configuration, metrics, and plots were generated successfully.

### Conclusion

The recommended model configuration fits in GPU memory with batch
size 8, and both training and validation loss decrease during the
20-step smoke test. The training and experiment-logging pipelines are
working correctly. This run is too short to assess final model quality.

### Artifacts

- [Configuration](runs/tinystories_memory_test_20260722_152311/config.json)
- [Metrics](runs/tinystories_memory_test_20260722_152311/metrics.csv)
- [Loss vs gradient steps](runs/tinystories_memory_test_20260722_152311/loss_vs_steps.png)
- [Loss vs wall-clock time](runs/tinystories_memory_test_20260722_152311/loss_vs_time.png)

### Next Step

Test larger batch sizes and then run a learning-rate sweep using a
larger token budget.