from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_mlp.config import TrainConfig

# Normalize with MNIST population mean and std
_MEAN = (0.1307,)
_STD  = (0.3081,)


def get_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Download MNIST (if needed) and return (train_loader, test_loader)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    print(
        f"[Data] Train samples: {len(train_dataset):,}  "
        f"Test samples: {len(test_dataset):,}  "
        f"Batch size: {cfg.batch_size}"
    )
    return train_loader, test_loader
