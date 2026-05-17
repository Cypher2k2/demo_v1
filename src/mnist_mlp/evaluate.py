import argparse

import torch

from mnist_mlp.config import TrainConfig
from mnist_mlp.data import get_dataloaders
from mnist_mlp.model import MNISTMLP

NUM_CLASSES = 10


def evaluate(cfg: TrainConfig) -> float:
    """Load the saved checkpoint and report test accuracy. Returns accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")
    print(f"[Evaluate] Loading checkpoint from {cfg.checkpoint_path}")

    model = MNISTMLP(cfg).to(device)
    state = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _, test_loader = get_dataloaders(cfg)

    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for pred, label in zip(preds, labels):
                class_correct[label] += int(pred == label)
                class_total[label]   += 1

    accuracy = correct / total
    print(f"\n[Evaluate] Per-class accuracy:")
    print(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*38}")
    for digit in range(NUM_CLASSES):
        digit_acc = class_correct[digit] / class_total[digit]
        print(
            f"  {digit:<8} {class_correct[digit]:>8} "
            f"{class_total[digit]:>8} {digit_acc:>9.2%}"
        )
    print(f"  {'-'*38}")
    print(f"  {'Overall':<8} {correct:>8} {total:>8} {accuracy:>9.2%}")
    print(f"\n[Evaluate] Test accuracy: {accuracy:.2%}  ({correct}/{total} correct)")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MNIST MLP checkpoint")
    parser.add_argument("--checkpoint", type=str, default="artifacts/mnist_mlp.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = TrainConfig(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    evaluate(cfg)


if __name__ == "__main__":
    main()
