"""mnist_mlp — simple MNIST MLP demo package for Cursor."""

__version__ = "0.1.0"

from mnist_mlp.config import TrainConfig
from mnist_mlp.model import MNISTMLP

__all__ = ["TrainConfig", "MNISTMLP", "__version__"]
