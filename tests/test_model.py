import torch
import pytest

from mnist_mlp.config import TrainConfig
from mnist_mlp.model import MNISTMLP


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig()


def test_mnist_mlp_output_shape(cfg):
    """Model should produce logits of shape (batch, 10) for any batch size."""
    model = MNISTMLP(cfg)
    model.eval()
    batch = torch.randn(32, 1, 28, 28)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"


def test_mnist_mlp_single_sample(cfg):
    """A single image should also produce shape (1, 10)."""
    model = MNISTMLP(cfg)
    model.eval()
    image = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(image)
    assert output.shape == (1, 10)


def test_forward_deterministic(cfg):
    """Given the same seed, two forward passes must produce identical logits."""
    model = MNISTMLP(cfg)
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(8, 1, 28, 28)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2), "Forward pass is not deterministic for identical input"


def test_config_defaults(cfg):
    """TrainConfig should expose expected default values."""
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 0.0
    assert cfg.step_size == 2
    assert cfg.lr_gamma == 0.5
    assert cfg.seed == 42
    assert cfg.epochs == 5
    assert cfg.batch_size == 64
    assert cfg.hidden1 == 256
    assert cfg.hidden2 == 128
