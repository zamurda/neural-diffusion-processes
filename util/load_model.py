from typing import Mapping, Union, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
import warnings
import sys
import pathlib

import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax

from ml_tools.checkpointing import TrainingState
from ml_tools import checkpointing

from neural_diffusion_processes.model import BiDimensionalAttentionModel

from .config_tools import NetworkConfig

def init_from_config(
    netwk_cfg: NetworkConfig, model: Union[BiDimensionalAttentionModel, Any]
    ) -> Tuple[Callable, Callable]:

    models_ = [
        BiDimensionalAttentionModel,
        Any
    ]

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        # TODO:
        # make a match-case with type(model) as the match and set attributes that way

        model = BiDimensionalAttentionModel(
            n_layers=netwk_cfg.n_layers,
            hidden_dim=netwk_cfg.hidden_dim,
            num_heads=netwk_cfg.num_heads,
        )
        return model(x, y, t, mask)

    @jax.jit
    def net(params, t, yt, x, mask, *, key):
        del key  # the network is deterministic
        #NOTE: Network awkwardly requires a batch dimension for the inputs
        return network.apply(params, t[None], yt[None], x[None], mask[None])[0]

    return net


def init_from_checkpoint(
    netwk_cfg: NetworkConfig,
    experiment_dir: Union[pathlib.Path, str],
    model: Union[BiDimensionalAttentionModel, Any],
    index: Optional[int] = -1
    ) -> Tuple[TrainingState, Callable]:
    '''
    Restores a training state from a checkpoint and initialises a model from it.
    Returns the TrainingState object (for further training) and the callable model
    '''

    state = TrainingState()
    if isinstance(experiment_dir, str):
        dir_ = pathlib.Path(experiment_dir)
    
    if not (dir_/'checkpoints').exists():
        raise ValueError(
            'Checkpoint does not exists in the experiment directory'
        )
    else:
        if index == -1: index = checkpointing.find_latest_checkpoint_step_index(str(dir_))
        else:
            if index is not None: index = int(index)
        state = checkpointing.load_checkpoint(state, str(dir_), step_index=int(index))
        print("Restored checkpoint at step {}".format(state.step))

        return state, init_from_config(netwk_cfg, model)