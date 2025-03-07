from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class Paths:
    train_folder: str
    train_target_folder: str
    val_folder: str
    val_target_folder: str


@dataclass
class Model:
    head: str
    backbone: str
    strategy: str


@dataclass
class Train:
    learning_rate: float
    optimizer: str
    batch_size: int
    num_workers: int


@dataclass
class Transform:
    size: int
    crop_factor: float
    blur_kernel_size: int
    p_color_jitter: float
    rand_rot: bool
    scaling: bool


@dataclass
class SoySegConfig:
    paths: Paths
    model: Model
    train: Train
    transform: Transform

# Register SoySegConfig
cs = ConfigStore.instance()
cs.store(name="soyseg_config", node=SoySegConfig)