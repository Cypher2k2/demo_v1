from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0
    step_size: int = 2
    lr_gamma: float = 0.5
    epochs: int = 5
    batch_size: int = 64

    # Architecture
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.2

    # Reproducibility
    seed: int = 42

    # Logging
    log_interval: int = 100  # print every N batches
    limit_batches: int = 0   # 0 = no limit (useful for quick smoke runs)

    # Paths
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    checkpoint_path: str = "artifacts/mnist_mlp.pt"
