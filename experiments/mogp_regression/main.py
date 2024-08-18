from typing import Mapping, Tuple, List, Union
from jaxtyping import Array, PyTree
import re
import os
import sys
import csv
import pprint
import string
import random
import datetime
import pathlib
import jax
import jax.numpy as jnp
import haiku as hk
from haiku.data_structures import partition, merge, traverse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import optax
import equinox as eqx
from functools import partial
from dataclasses import asdict
from data import gen_dataset, se_kernel, _MOGP_SAMPLES_PER_EPOCH
import matplotlib.pyplot as plt
from einops import rearrange

# from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools.writers import AimWriter
from ml_tools import actions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.types import Dataset, Batch , Rng
from neural_diffusion_processes.multichannel import MultiChannelEncodingModel
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion

from absl import flags # for training on apple gpu
flags.DEFINE_bool('metal', False, "whether gpu training should occur")

from util.config_tools import get_config_map, parse_config_map, dict_to_yaml, Config, DatasetConfig
from util.load_model import init_state_from_pickle

jax.config.update('jax_disable_jit', False) # set true for debugging

MASK_TYPE_CAUSAL = jnp.array([])
MASK_TYPE_NOMASK = jnp.array([[]])

config_map: Mapping = get_config_map()
config: Config = parse_config_map(config_map)
key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)

PRETRAINED_MODEL_DIR = pathlib.Path(config.restore.checkpoint_dir)
PRETRAINED_MODEL_NAME = PRETRAINED_MODEL_DIR.name

timestamp = datetime.datetime.now().strftime("%b%d")
letters = string.ascii_lowercase
id = ''.join(random.choice(letters) for i in range(4))
EXPERIMENTS_DIR = './trained_models'
EXPERIMENT = f'mogp_{timestamp}_{id}_{PRETRAINED_MODEL_NAME}'

SAVE_HERE = pathlib.Path(EXPERIMENTS_DIR)/pathlib.Path(EXPERIMENT)
if not SAVE_HERE.exists():
    os.mkdir(SAVE_HERE)

# generate MOGP data
kernels = {'se': partial(se_kernel, sigma2=1, l=0.35)}
coreg_weights = jnp.array([
    [0.5, 0.5],
    [0.2, 0.8],
    [0.6, 0.4],
    [0.1, 0.9]
])
ds_train = gen_dataset(
    config.seed,
    kernels[config.dataset.data],
    coreg_weights,
    -1,
    1,
    config.training.num_epochs,
    config.training.batch_size,
    config.dataset.sample_length
)

# init optimizer
NUM_STEPS =  (_MOGP_SAMPLES_PER_EPOCH // (32 // coreg_weights.shape[0])) * config.training.num_epochs
steps_per_epoch = NUM_STEPS // config.training.num_epochs
learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=config.optimizer.init_lr,
    peak_value=config.optimizer.peak_lr,
    warmup_steps=steps_per_epoch * config.optimizer.num_warmup_epochs,
    decay_steps=steps_per_epoch * config.optimizer.num_decay_epochs,
    end_value=config.optimizer.end_lr,
)

# purpose of label function is to assign 'update' or 'fix' to parameter names for the multi_transform to the updates
def label_fn(ptree):
    labeller = lambda path, _: 'fix' if bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_model\b(?:\/\w+)*', '/'.join(str(p) for p in path))) else 'update'
    return jax.tree_util.tree_map_with_path(labeller, ptree)

update_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)
optimizer = optax.multi_transform(
    {"update": update_chain, "fix": optax.set_to_zero()}, param_labels=label_fn
)

# define network functions
@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask_type):
    model = MultiChannelEncodingModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
        n_blocks=4,
        n_channels=coreg_weights.shape[0]
    )
    return model(x, y, t, mask_type)
    # return partial(model, mask_type=mask_type)(x, y, t)


@jax.jit
def net(params, t, yt, x, mask_type, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs
    out = network.apply(params, t, yt, x, mask_type)[0]
    W, b = params['multi_channel_encoding_model/bi_dimensional_attention_model/linear_2']['w'], params['multi_channel_encoding_model/bi_dimensional_attention_model/linear_2']['b']
    return jnp.einsum('...nh,h1 -> ...n1', out, W) + b


# loss function with trainable and fixed weights:
def loss_fn(params_trainable, params_fixed, batch, key) -> jnp.ndarray:
    params = merge(params_trainable, params_fixed)
    net_with_params = partial(net, params)
    kwargs = dict(
        num_timesteps=config.diffusion.timesteps,
        loss_type=config.optimizer.loss_type,
        mask_type=MASK_TYPE_NOMASK,
        n_channels=coreg_weights.shape[0]
        )
    return ndp.process.loss_multichannel(process, net_with_params, batch, key, **kwargs)

def loss_fn(params, batch, key) -> jnp.ndarray:
    net_with_params = partial(net, params)
    kwargs = dict(
        num_timesteps=config.diffusion.timesteps,
        loss_type=config.optimizer.loss_type,
        mask_type=MASK_TYPE_NOMASK,
        n_channels=coreg_weights.shape[0]
        )
    return ndp.process.loss_multichannel(process, net_with_params, batch, key, **kwargs)


@jax.jit
def ema_update(decay, ema_params, new_params):
    predicate = lambda m, n, v: bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_model\b(?:\/\w+)*', m))
    trbl_params, fix_params = partition(predicate, new_params)
    trbl_ema, fix_ema = partition(predicate, ema_params)
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params
    return merge(jax.tree_map(_ema, trbl_ema, trbl_params), fix_ema) #changed from tree.map since jax has been downgraded


# update which only does wrt trainable params:
@jax.jit
def update_step(state: TrainingState, batch: Batch) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    # trainable, fixed = partition(lambda m, n, v: bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_model\b(?:\/\w+)*', m)), state.params)
    loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    new_params_ema = ema_update(config.optimizer.ema_rate, state.params_ema, new_params)
    new_state = TrainingState(
        params=new_params,
        params_ema=new_params_ema,
        opt_state=new_opt_state,
        key=new_key,
        step=state.step + 1
    )
    metrics = {
        'loss': loss_value,
        'step': state.step
    }
    return new_state, metrics

@jax.jit
def sample_prior(key, x, model_fn, mask_type):
    return process.sample(key, x, mask_type, model_fn=model_fn)

@partial(jax.vmap, in_axes=(0,None,None,None))
def sample_n_priors(keys, x, mask_type):
    model_fn = partial(net, state.params_ema)
    return sample_prior(keys, x, model_fn, mask_type)


# prior and process plots
def prior_plots(batch, state) -> None:
    # n_channels = coreg_weights.shape[0]
    # chosen_sample = jax.random.choice(jax.random.seed(53), jnp.arange(n_channels))
    # x = rearrange(batch.x_target, '(b c) n d -> b c n d', c=n_channels)[chosen_sample]
    # y = rearrange(batch.y_target, '(b c) n 1 -> b c n 1', c=n_channels)[chosen_sample]
    # samples = sample_n_priors(jax.random.split(jax.random.key(42)), x, mask_type=MASK_TYPE_NOMASK)
    # fig, ax = plt.subplots(n_channels, 2)
    # for i in range(n_channels):
    #     x = x[i].squeeze(-1)
    #     args = x.argsort()
    #     ax[i,0].plot(x[args], samples[...,0])
    fig,ax = plt.subplots()
    return fig

def process_plots(batch, state) -> None:
    fig, ax = plt.subplots()
    return fig

# load pretrained checkpoint
pretrained_state: TrainingState = init_state_from_pickle(PRETRAINED_MODEL_DIR)

def traverse_and_switch(*, to_place, place_in):
    for m, n, v in traverse(place_in):
        if bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_model\b(?:\/\w+)*', m)):
            name = re.sub(r'^(?:\w+\/)*\bbi_dimensional_attention_model\b', 'bi_dimensional_attention_model', m)
            v = to_place[name][n]
        else:
            v = v
        # place back in
        place_in[m][n] = v

@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask_type=MASK_TYPE_NOMASK
    )
    traverse_and_switch(to_place=pretrained_state.params_ema, place_in=initial_params)
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )

# init state
INPUT_DIM = 1
batch_init = Batch(
    x_target=jnp.zeros((32, config.dataset.sample_length, INPUT_DIM)),
    y_target=jnp.zeros((32, config.dataset.sample_length, 1)),
)
state: TrainingState = init(batch_init, jax.random.PRNGKey(config.seed))

aimwriter = AimWriter(EXPERIMENT)
cfg_dict = asdict(config)
cfg_dict['mask_type'] = 'no mask'
cfg_dict['coreg_weights'] = coreg_weights.tolist()
aimwriter.log_hparams(cfg_dict)


actions = [
    actions.PeriodicCallback(
        every_steps=1,
        callback_fn=lambda step, t, **kwargs: aimwriter.write_scalars(step, kwargs["metrics"])
    ),
    actions.PeriodicCallback(
        every_steps=1,
        callback_fn=lambda step, t, **kwargs: state_utils.save_checkpoint(kwargs["state"], SAVE_HERE, step)
    ),
    actions.PeriodicCallback(
        every_steps=2*steps_per_epoch,
        callback_fn=lambda step, t, **kwargs: aimwriter.write_figures(
            step,
            {
                'prior samples':prior_plots(batch=kwargs['batch'], state=kwargs['state']),
                'predicted noise':process_plots(batch=kwargs['batch'], state=kwargs['state'])
            }
        )
    ),
    actions.PeriodicCallback(
        every_steps=1,
        callback_fn=lambda step, t, **kwargs: kwargs['writer'].writerow([step, kwargs['metrics']['loss'], kwargs['metrics']['lr']])
    )
]

steps = range(state.step + 1, NUM_STEPS + 1)
progress_bar = tqdm.tqdm(steps)

def check_if_fixed(predicate, params_old, params_new) -> bool:
    """just for testing if fixed params are actually fixed"""
    old_fixed, _ = partition(predicate, params_old)
    new_fixed, __ = partition(predicate, params_new)
    return eqx.tree_equal(old_fixed, new_fixed)

with open(SAVE_HERE/'metrics.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['step', 'loss', 'learning_rate'])

    for step, batch in zip(progress_bar, ds_train):
        if step >= 5: break
        if step < state.step: continue  # wait for the state to catch up in case of restarts
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(state.step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key, batch=batch, writer=csvwriter)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")

print(f'============================ {EXPERIMENT} finished training ===============================')
dict_to_yaml(cfg_dict, SAVE_HERE/'config.yaml')