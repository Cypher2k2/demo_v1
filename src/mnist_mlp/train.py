import argparse
import os

import torch
import torch.nn as nn

from mnist_mlp.config import TrainConfig
from mnist_mlp.data import get_dataloaders
from mnist_mlp.model import MNISTMLP


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    train_loader, test_loader = get_dataloaders(cfg)

    model = MNISTMLP(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_batches = (
        min(cfg.limit_batches, len(train_loader))
        if cfg.limit_batches > 0
        else len(train_loader)
    )

    print(
        f"[Train] Starting training for {cfg.epochs} epoch(s)  "
        f"lr={cfg.lr}  weight_decay={cfg.weight_decay}  batch_size={cfg.batch_size}"
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            if cfg.limit_batches > 0 and batch_idx > cfg.limit_batches:
                break

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % cfg.log_interval == 0:
                print(
                    f"[Train] Epoch {epoch}/{cfg.epochs}  "
                    f"batch {batch_idx:4d}/{total_batches}  "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = running_loss / total_batches
        accuracy = correct / total
        print(
            f"[Epoch {epoch}] avg_loss={avg_loss:.4f}  "
            f"accuracy={accuracy:.2%}  lr={cfg.lr}"
        )

    torch.save(model.state_dict(), cfg.checkpoint_path)
    print(f"[Train] Checkpoint saved to {cfg.checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP")
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=0.0)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--hidden1",       type=int,   default=256)
    parser.add_argument("--hidden2",       type=int,   default=128)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--log-interval",  type=int,   default=100)
    parser.add_argument("--limit-batches", type=int,   default=0,
                        help="Stop after N batches per epoch (0 = no limit)")

    args = parser.parse_args()
    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        log_interval=args.log_interval,
        limit_batches=args.limit_batches,
    )
    train(cfg)


if __name__ == "__main__":
    main()
