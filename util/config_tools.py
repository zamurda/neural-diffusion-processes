from typing import Mapping, Union, Any, Optional, List
from dataclasses import dataclass, field
import warnings
import sys
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

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 250


@dataclass
class Config:
    seed: int = 53
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    schedule: DiffusionConfig = field(default_factory=DiffusionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    restore: str = ""
     # remember to calculate this in training loop
    '''@property
    def steps_per_epoch(self) -> int:
        return self.samples_per_epoch // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.num_epochs'''


import tomli # use instead of tomllib as we are working in python 3.10
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('config', None, 'path to config.toml file')

def configtoml_to_map(path: Union[str, pathlib.Path]) -> Mapping:

    config_dir = pathlib.Path(path)
    if config_dir.exists():
        with open(config_dir, mode='rb') as cfg:
            config = tomli.load(cfg)
    else:
        raise ValueError(
            f'Config file at location {path} does not exist'
        )
    return config

def get_config_map() -> Mapping:
    flags.FLAGS(sys.argv)
    if not FLAGS.config:
        raise RuntimeError(
            'path to config file not supplied. model cannot be initialised'
        )
    path = FLAGS.config
    return configtoml_to_map(path)

def parse_config_map(config_map: Mapping) -> Config:
    configs = {
        'DiffusionConfig': DiffusionConfig(),
        'OptimizerConfig': OptimizerConfig(),
        'NetworkConfig': NetworkConfig(),
        'DatasetConfig': DatasetConfig(),
        'TrainingConfig': TrainingConfig()
    }

    config = Config()

    for table_name in configs.keys():
        if not config_map.get(table_name, False):
            if table_name == 'DatasetConfig':
                raise ValueError(
                    'DatasetConfig not present in .toml file'
                )
            else: warnings.warn(
                f'config table not present in toml file, reverting to defaults for {table_name}',
                RuntimeWarning
            )
        else:
            # set the values for that map to the attributes of the config class
            cfg_class = configs[table_name]
            for attr in config_map[table_name].keys():
                setattr(cfg_class, attr, config_map[table_name][attr])
            setattr(config, table_name, cfg_class)
    return config
