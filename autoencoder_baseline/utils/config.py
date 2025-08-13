# config.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import yaml
import torch
from dacite import from_dict, Config as DaciteConfig


@dataclass
class ModelConfig:
    encoder_widths: Tuple[int, ...]
    latent_dim: int
    data_shape: int = field(init=False)


@dataclass
class TrainingConfig:
    lr: float = 1e-5
    batch_size: int = 16
    n_epochs: int = 500
    save_ckpt_step: int = 500
    device: str = "cuda"


@dataclass
class PathConfig:
    data_dir: Path
    checkpoints_dir: Path
    figs_dir: Path


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig
    experiment_name: str

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Configure dacite for type conversions
        dacite_config = DaciteConfig(
            type_hooks={
                Path: lambda x: Path(x) if isinstance(x, str) else x,
                Tuple[int, ...]: lambda x: tuple(x) if isinstance(x, list) else x,
                torch.device: lambda s: torch.device(s) if isinstance(s, str) else s,
                float: lambda s: float(s) if isinstance(s, str) else s,
                int: lambda x: int(float(x)) if isinstance(x, (str, float)) else x,
            }
        )

        return from_dict(data_class=cls, data=config_dict, config=dacite_config)
