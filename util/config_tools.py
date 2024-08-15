from typing import Mapping, Union, Any, Optional, List
from dataclasses import dataclass, field
import warnings
import sys
import pathlib
import yaml
import tomli # use instead of tomllib as we are working in python 3.10
from absl import flags

from ._config_defaults import (
    DiffusionConfig,
    OptimizerConfig,
    DatasetConfig,
    TrainingConfig,
    NetworkConfig,
    RestoreConfig,
    Config
)

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
        'diffusion': DiffusionConfig(),
        'optimizer': OptimizerConfig(),
        'network': NetworkConfig(),
        'dataset': DatasetConfig(),
        'training': TrainingConfig(),
        'restore': RestoreConfig()
    }

    config = Config()

    for table_name in configs.keys():
        if not config_map.get(table_name, False):
            if table_name == 'dataset':
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


def dict_to_yaml(dictionary: dict, file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    :param dictionary: The dictionary to save.
    :param file_path: The path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(dictionary, file)

def yaml_to_dict(file_path:str) -> dict:
    """
    Load a dictionary from a YAML file.

    :param file_path: The path to the YAML file.
    :return: The dictionary loaded from the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
