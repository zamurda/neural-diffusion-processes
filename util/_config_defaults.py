from typing import Mapping, Union, Any, Optional, List
from dataclasses import dataclass, field
import pathlib

@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class OptimizerConfig:
    loss_type: str = 'l1'
    num_warmup_epochs: int = 20
    num_decay_epochs: int = 200
    init_lr: float = 2e-5
    peak_lr: float = 1e-3
    end_lr: float = 1e-5
    ema_rate: float = 0.995  # 0.999


@dataclass
class NetworkConfig:
    n_layers: int = 4
    hidden_dim: int = 64
    num_heads: int = 8


@dataclass
class EvalConfig:
    batch_size: int = 4
    float64: bool = False

@dataclass
class DatasetConfig:
    data: Union[pathlib.Path, str] = None
    features: List[str] = None
    target_index: str = None
    sample_length: int = 50 # how big should each sampled function draw be?

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 250

@dataclass
class RestoreConfig:
    checkpoint_dir: str = ''
    from_index: int = -1


@dataclass
class Config:
    seed: int = 53
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    schedule: DiffusionConfig = field(default_factory=DiffusionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    restore: RestoreConfig = field(default_factory=RestoreConfig)

    # restore: str = ""
     # remember to calculate this in training loop
    '''@property
    def steps_per_epoch(self) -> int:
        return self.samples_per_epoch // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.num_epochs'''