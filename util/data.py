"""functions which return dataset generators to iterate over"""

import pathlib
from typing import Union, Optional, Tuple, Mapping, List
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from einops import rearrange

from .config_tools import Config, DatasetConfig, TrainingConfig
from neural_diffusion_processes.types import Batch, Dataset


def _make_batch(data: Mapping, colnames: List, target_name: str, indices: jnp.array) -> Batch:
    # num_points_bool = jnp.ones(data.shape[0], dtype=bool)
    features_arr = []
    for c in colnames:
        features_arr.append(jnp.take(jnp.array(data[c]), indices))
    target = jnp.take(jnp.array(data[target_name]), indices)
    features = jnp.stack(features_arr)
    features = rearrange(features, "input_dim batch_size seq_len -> batch_size seq_len input_dim")

    return Batch(x_target=features, y_target=target[..., None])


def gen_dataset_standard(
    dir_: Union[str, pathlib.Path],
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    seed: int = 53,
    target_is_normalised: Optional[bool] = True,
    sample_contiguous: Optional[bool] = False,
) -> Tuple[Dataset, int, Mapping]:
    """
    Made to read pickled pandas dataframes or CSV files containing samples.
    If data is time-series, then the dataframe must be ORDERED.

    Returns:
    Dataset - Generator : which can be iterated over in the training loop
    Num_steps - int     : total number of training steps
    extras - dict       : dict containing the mean and standard deviation of the target variable (if not already normalised)
    """
    # check if data exists
    dir_ = dir_ if isinstance(dir_, pathlib.Path) else pathlib.Path(dir_)
    name = pathlib.Path(dataset_config.data)
    if (dir_ / name).exists():
        file = dir_ / name

        # load it in
        match file.suffix:
            case ".pkl":
                data: Mapping = jnp.load(file, allow_pickle=True)
            case ".csv":
                data: Mapping = pd.read_csv(file)

        # pull required columns
        if dataset_config.features is not None and dataset_config.target_index is not None:
            features_set = set(dataset_config.features)
            features_set.add(dataset_config.target_index)
            if not features_set.issubset(set(data.keys())):
                raise ValueError(
                    f"one or more of the features or the target are not present in the dataset"
                )
        else:
            raise ValueError(f"feature names or target have not been provided")

        # repeat and batch it up
        num_samples = data.shape[0]  # dataset size
        indices = jnp.arange(num_samples)

        key = jax.random.PRNGKey(seed)
        batched_indices = []
        num_batches = (num_samples * training_config.num_epochs) // training_config.batch_size

        """for _ in range(num_batches):
            curr_batch = []
            for __ in range(training_config.batch_size):
                key, subkey = jax.random.split(key)
                # sort the indices so that causal masking can be applied
                # without replacement since sample_length << num_samples almost always
                curr_batch.append(
                    jax.random.choice(subkey, indices, (dataset_config.sample_length,))
                )
            batched_indices.append(jnp.stack(curr_batch))"""

        def get_sample_indices(key, indices, batch_size, sample_length):
            return jax.random.choice(
                key,
                indices,
                (
                    batch_size,
                    sample_length,
                ),
                replace=True,
            )

        keys = jax.random.split(key, num_batches)
        batched_indices = jax.vmap(get_sample_indices, in_axes=(0, None, None, None))(
            keys, indices, training_config.batch_size, dataset_config.sample_length
        )

        if not target_is_normalised:
            mean = data[dataset_config.target_index].mean()
            std = data[dataset_config.target_index].std()
            data[dataset_config.target_index] = (data[dataset_config.target_index] - mean) / std

        else:
            mean, std = 0, 0

        jitted_make_batch = jax.jit(
            partial(_make_batch, data, dataset_config.features, dataset_config.target_index)
        )

        return (
            map(jitted_make_batch, batched_indices),
            (num_samples * training_config.num_epochs) // training_config.batch_size,
            {"mean": mean, "std": std},
        )

    else:
        raise ValueError(f"dataset {name} does not exist in {dir_}")


def gen_batch_eval(
    dir_: Union[str, pathlib.Path],
    dataset_config: DatasetConfig,
    seed: Optional[int] = 53,
    n_draws: Optional[int] = 10,
    make_context: Optional[bool] = True,
    num_context_points: int = None,
    target_is_normalised: Optional[bool] = True,
    context_type: Optional[str] = "random",
) -> Batch:
    key = jax.random.key(seed)
    dir_ = dir_ if isinstance(dir_, pathlib.Path) else pathlib.Path(dir_)
    name = pathlib.Path(dataset_config.data)
    if (dir_ / name).exists():
        file = dir_ / name

        # load it in
        match file.suffix:
            case ".pkl":
                data: Mapping = jnp.load(file, allow_pickle=True)
            case ".csv":
                data: Mapping = pd.read_csv(file)

        # pull required columns
        if dataset_config.features is not None and dataset_config.target_index is not None:
            features_set = set(dataset_config.features)
            features_set.add(dataset_config.target_index)
            if not features_set.issubset(set(data.keys())):
                raise ValueError(
                    f"one or more of the features or the target are not present in the dataset"
                )
        else:
            raise ValueError(f"feature names or target have not been provided")

        # don't use (yet)
        if not target_is_normalised:
            mean = data[dataset_config.target_index].mean()
            std = data[dataset_config.target_index].std()
            data[dataset_config.target_index] = (data[dataset_config.target_index] - mean) / std

        else:
            mean, std = 0, 0

        # get indices
        num_samples = data.shape[0]
        indices = jnp.arange(num_samples)

        to_choose = jax.random.choice(key, indices, shape=(n_draws, dataset_config.sample_length))
        batch_without_ctx = _make_batch(
            data, dataset_config.features, dataset_config.target_index, indices=to_choose
        )

        num_context_points = (
            dataset_config.sample_length // 4 if num_context_points is None else num_context_points
        )  # choose 1/4 of the sample length to be context by default
        if make_context:
            match context_type.lower():
                case "random":
                    to_take = []
                    for _ in range(n_draws):
                        key, subkey = jax.random.split(key)
                        to_take.append(
                            jax.random.choice(
                                subkey,
                                jnp.arange(dataset_config.sample_length),
                                shape=(num_context_points,),
                                replace=False,
                            )
                        )
                    to_take = jnp.stack(to_take)
                    x_c = []
                    x_t = []
                    y_c = []
                    y_t = []
                    for i in range(n_draws):
                        x, y = (
                            batch_without_ctx.x_target[i, :, :],
                            batch_without_ctx.y_target[i, :, :],
                        )
                        # print(to_take[i])
                        # print(x)
                        # print(y)
                        x_c.append(x[to_take[i], :])
                        y_c.append(y[to_take[i], :])
                        mask = (
                            jnp.ones(dataset_config.sample_length, dtype=bool)
                            .at[to_take[i]]
                            .set(False)
                            .reshape(x.shape[0], 1)
                        )
                        mask_x = jnp.tile(mask, (1, x.shape[-1]))
                        mask_y = jnp.tile(mask, (1, y.shape[-1]))
                        # print(x[mask_x].shape)
                        # print(y[mask_y].shape)
                        # print('\n\n')
                        x_t.append(
                            x[mask_x].reshape(
                                dataset_config.sample_length - num_context_points, x.shape[-1]
                            )
                        )
                        y_t.append(
                            y[mask_y].reshape(
                                dataset_config.sample_length - num_context_points, y.shape[-1]
                            )
                        )
                    x_c = jnp.stack(x_c)
                    y_c = jnp.stack(y_c)
                    x_t = jnp.stack(x_t)
                    y_t = jnp.stack(y_t)
                    batch_with_ctx = Batch(x_context=x_c, x_target=x_t, y_context=y_c, y_target=y_t)

                case "causal":
                    raise NotImplementedError
                case _:
                    raise ValueError(f"context type {context_type} is not valid")

            return batch_with_ctx

        else:
            return batch_without_ctx
