''' functions which return dataset generators to iterate over'''

import pathlib
from typing import Union, Optional, Tuple, Mapping, List
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd

from .config_tools import Config, DatasetConfig, TrainingConfig
from neural_diffusion_processes.types import Batch, Dataset


def _make_batch(data: Mapping,
                colnames: List,
                target_name: str,
                indices: jnp.array
            ) -> Batch:
    
    features_arr = []
    for c in colnames:
        feature = jnp.take(jnp.array(data[c]), indices)
        features_arr.append(feature)
    target = jnp.take(jnp.array(data[target_name]), indices)
    features = jnp.stack(features_arr)

    return Batch(x_target=features[...,None].squeeze(0), y_target=target[...,None]) #NOTE: for some reason features has an extra preciding dimension of length 1



def gen_dataset_standard(
        dir_: Union[str, pathlib.Path],
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        seed: int = 53,
        drop_remainder: bool = True
    ) -> Tuple[Dataset, int]:
    '''
    Made to read pickled pandas dataframes or CSV files containing samples.
    If data is time-series, then the dataframe must be ORDERED.

    Returns:
    Dataset - Generator : which can be iterated over in the training loop
    Num_steps - int     : total number of training steps
    '''
    # check if data exists
    dir_ = dir_ if isinstance(dir_, pathlib.Path) else pathlib.Path(dir_)
    name = pathlib.Path(dataset_config.data)
    if (dir_/name).exists():
        file = (dir_/name)

        # load it in
        match file.suffix:
            case '.pkl':
                data: Mapping = jnp.load(file, allow_pickle=True)
            case '.csv':
                data: Mapping = pd.read_csv(file)
        
        # pull required columns
        if dataset_config.features is not None and dataset_config.target_index is not None:
            features_set = set(dataset_config.features)
            features_set.add(dataset_config.target_index)
            if not features_set.issubset(set(data.keys())):
                raise ValueError(f'one or more of the features or the target are not present in the dataset')
        else: raise ValueError(f'feature names or target have not been provided')

        # repeat and batch it up
        num_samples = data.shape[0] # dataset size
        indices = jnp.arange(num_samples)

        key = jax.random.PRNGKey(seed)
        batched_indices = []
        for _ in range((num_samples*training_config.num_epochs)//training_config.batch_size):
            key, subkey = jax.random.split(key)
            # sort the indices so that causal masking can be applied
            batched_indices.append(
                jnp.sort(jax.random.choice(
                    subkey, indices, (training_config.batch_size, dataset_config.sample_length)
                        )
                    )
                )
        jitted_make_batch = jax.jit(
            partial(_make_batch, data, dataset_config.features, dataset_config.target_index)
            )
        
        return map(jitted_make_batch, batched_indices), (num_samples*training_config.num_epochs)//training_config.batch_size
        
    else: raise ValueError(f'dataset {name} does not exist in {dir_}')