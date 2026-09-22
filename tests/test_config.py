from pathlib import Path

from fedgagan.config import load_config


def test_demo_config_resolves_paths():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "demo.yaml")
    assert config.model.latent_dim == 32
    assert Path(config.data.paths[0]).is_absolute()
    assert Path(config.output_dir).is_absolute()

