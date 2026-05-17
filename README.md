# MNIST Cursor Demo Codebase

A small, readable PyTorch project built to demo Cursor workflows.
It trains and evaluates a simple MLP on MNIST with clear logs, CLI entry points,
and a tiny test suite — easy to live-code against.

## Project layout

```
src/mnist_mlp/
  config.py     — TrainConfig dataclass (all hyperparameters in one place)
  model.py      — MNISTMLP network  (784 → 256 → 128 → 10)
  data.py       — MNIST download and dataloader setup
  train.py      — Training loop, metrics, checkpoint save
  evaluate.py   — Checkpoint load and test accuracy report
tests/
  test_model.py — Shape + determinism checks
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package with dev dependencies
pip install -e ".[dev]"
```

## Run the demo

Fast smoke run (1 epoch, 10 batches):

```bash
mnist-train --epochs 1 --limit-batches 10 --log-interval 2
```

Full training run:

```bash
mnist-train --epochs 5
```

Evaluate the saved checkpoint:

```bash
mnist-evaluate
```

Run tests:

```bash
python -m pytest -v
```

## Example output

```
[Train] Starting training for 5 epochs  lr=0.001  batch_size=64
[Train] Epoch 1/5  batch  10/938  loss=2.2987
[Train] Epoch 1/5  batch 100/938  loss=0.4312
...
[Epoch 1] avg_loss=0.3841  accuracy=88.76%  lr=0.001
[Evaluate] Loading checkpoint from artifacts/mnist_mlp.pt
[Evaluate] Test accuracy: 97.43%  (9743/10000 correct)
```

## Cursor demo prompts

Try these prompts in Cursor to show AI-assisted coding:

1. **Add a confusion matrix** — "Add a confusion matrix print after evaluation showing predicted vs actual for each digit."
2. **Refactor logging** — "Refactor the batch log print into a reusable `log_batch` helper function."
3. **Add weight decay** — "Add a `--weight-decay` CLI arg to `train.py` and wire it into the Adam optimizer."
4. **Seed test** — "Write a pytest test that checks the model output is identical for two forward passes with the same seed."
5. **Learning rate scheduler** — "Add a StepLR scheduler that halves the learning rate every 2 epochs."
6. **Per-class accuracy** — "Print a per-class accuracy table in `evaluate.py` showing accuracy for each digit 0–9."

## Notes

- MNIST downloads automatically to `data/` on first run.
- Checkpoints are saved to `artifacts/mnist_mlp.pt`.
- Code favors readability over abstraction — ideal for live demos.
