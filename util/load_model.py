from typing import Mapping, Union, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
import warnings
import sys
import pathlib
import pickle

import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax

from ml_tools.state_utils import TrainingState
from ml_tools import state_utils

from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.multichannel import MultiChannelBDAM

from .config_tools import NetworkConfig

def init_from_config(
    netwk_cfg: NetworkConfig, model: Union[BiDimensionalAttentionModel, Any]
    ) -> Callable:

    models_ = [
        BiDimensionalAttentionModel,
        Any
    ]

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask_type):
        # TODO:
        # make a match-case with type(model) as the match and set attributes that way

        model = BiDimensionalAttentionModel(
            n_layers=netwk_cfg.n_layers,
            hidden_dim=netwk_cfg.hidden_dim,
            num_heads=netwk_cfg.num_heads,
        )
        return model(x, y, t, mask_type)

    def net(params, t, yt, x, mask_type, *, key):
        del key  # the network is deterministic
        return network.apply(params, t, yt, x, mask_type)[0]

    return net


def restore_from_checkpoint(
    netwk_cfg: NetworkConfig,
    experiment_dir: Union[pathlib.Path, str],
    model: Union[BiDimensionalAttentionModel, Any],
    index: Optional[int] = -1
    ) -> Tuple[TrainingState, Callable]:
    '''
    Restores a training state from a checkpoint and initialises a model from it.
    Returns the TrainingState object (for further training) and the callable model
    '''

    if isinstance(experiment_dir, str):
        dir_ = pathlib.Path(experiment_dir)
    
    if not (dir_/'checkpoints').exists():
        raise ValueError(
            'Checkpoint does not exists in the experiment directory'
        )
    else:
        if index == -1: index = state_utils.find_latest_checkpoint_step_index(str(dir_))
        else:
            if index is not None: index = int(index)
        state = init_state_from_pickle(dir_)
        state = state_utils.load_checkpoint(state, str(dir_), step_index=int(index))
        print("Restored checkpoint at step {}".format(state.step))

        return state, init_from_config(netwk_cfg, model)
    
def init_state_from_pickle(experiment_name: str) -> TrainingState:
    p = pathlib.Path(experiment_name)
    with open(p/'latest_trainingstate.pkl', 'rb') as file:
        try:
            state = pickle.load(file)
        except FileNotFoundError:  raise FileNotFoundError(f"The file 'latest_trainingstate.pkl' does not exist in the directory '{experiment_name}'.")
        except pickle.UnpicklingError: raise ValueError("The file could not be unpickled")
    return state
