from typing import Mapping, Tuple, List, Union
from jaxtyping import Array, PyTree
import pickle
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
from data import gen_dataset, make_batch, se_kernel, _MOGP_SAMPLES_PER_EPOCH
import matplotlib.pyplot as plt
from einops import rearrange

# from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools.writers import AimWriter
from ml_tools import actions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.types import Dataset, Batch , Rng
from neural_diffusion_processes.multichannel import MultiChannelBDAM
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion

from absl import flags # for training on apple gpu
flags.DEFINE_bool('metal', False, "whether gpu training should occur")
flags.DEFINE_bool('nvidia', False, "whether jax should look for NVIDIA gpus")

from util.config_tools import get_config_map, parse_config_map, dict_to_yaml, Config, DatasetConfig
from util.load_model import init_state_from_pickle

jax.config.update('jax_disable_jit', False) # set true for debugging
FLAGS = flags.FLAGS
flags.FLAGS(sys.argv)
if flags.FLAGS.metal or flags.FLAGS.nvidia:
    os.environ['JAX_PLATFORMS'] = 'gpu'
else: os.environ['JAX_PLATFORMS'] = 'cpu'

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
EXPERIMENT = f'mogp_{timestamp}_{id}'

SAVE_HERE = pathlib.Path(EXPERIMENTS_DIR)/pathlib.Path(EXPERIMENT)
if not SAVE_HERE.exists():
    os.makedirs(SAVE_HERE, exist_ok=True)

class Params(eqx.Module):
    params: PyTree
    params_ema: PyTree
    step: int

# generate MOGP data
kernels = {'se': partial(se_kernel, sigma2=1, l=0.35)}
num_channels = 4
num_latents = 2
weightskey, datasetkey = jax.random.split(key)
coreg_weights = jax.random.normal(weightskey, (num_channels, num_latents))
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
NUM_STEPS =  (_MOGP_SAMPLES_PER_EPOCH // config.training.batch_size) * config.training.num_epochs
steps_per_epoch = NUM_STEPS // config.training.num_epochs
# learning_rate_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=config.optimizer.init_lr,
#     peak_value=config.optimizer.peak_lr,
#     warmup_steps=steps_per_epoch * config.optimizer.num_warmup_epochs,
#     decay_steps=steps_per_epoch * config.optimizer.num_decay_epochs,
#     end_value=config.optimizer.end_lr,
# )
learning_rate_schedule = optax.constant_schedule(3e-4)

# purpose of label function is to assign 0 or 1 to parameter names for the multi_transform to the updates
def label_fn(ptree):
    labeller = lambda path, _: 1 if bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_block[_\d+]*\b(?:\/\w+)*', '/'.join(str(p) for p in path))) or bool(re.search('multi_channel_bdam/linear(?!_2)', '/'.join(str(p) for p in path))) else 1
    return jax.tree_util.tree_map_with_path(labeller, ptree)

update_chain = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)
optimizer = update_chain

@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask_type):
    model = MultiChannelBDAM(
        n_layers=config.network.n_layers,
        n_channels=num_channels,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
        init_zero=True,
    )
    return model(x, y, t, mask_type)

@jax.jit
def net(params, t, yt, x, mask_type, *, key):
    del key  # the network is deterministic
    #NOTE: No batch dimension for network inputs 
    out = network.apply(params, t, yt, x, mask_type)
    return out


def loss_fn(params, batch, key) -> jnp.ndarray:
    net_with_params = partial(net, params)
    kwargs = dict(
        num_timesteps=config.diffusion.timesteps,
        loss_type=config.optimizer.loss_type,
        mask_type=MASK_TYPE_NOMASK,
        n_channels=coreg_weights.shape[0]
        )
    return ndp.process.loss_multichannel(process, net_with_params, batch, key, **kwargs)


# @jax.jit # fix or update
# def ema_update(decay, labels, ema_params, new_params):
#     def _ema(ema_params, new_params, label):
#         return jax.lax.cond(
#             label == 1,
#             lambda: decay * ema_params + (1.0 - decay) * new_params,
#             lambda: ema_params
#         )
#     return jax.tree.map(_ema, ema_params, new_params, labels) #jax upgraded

@jax.jit
def ema_update(decay, ema_params, new_params):
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params
    return jax.tree.map(_ema, ema_params, new_params)


# update which only does wrt trainable params:
@jax.jit
def update_step(state: TrainingState, batch: Batch) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    # trainable, fixed = partition(lambda m, n, v: bool(re.search(r'(?:\w+/)*\bbi_dimensional_attention_model\b(?:\/\w+)*', m)), state.params)
    loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    # labels = label_fn(state.params)
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

def sample_prior(key, x, mask_type, model_fn):
    return process.sample(key, x, mask_type, model_fn=model_fn, batched_input=True)

@partial(jax.vmap, in_axes=(0,None,None,None))
def sample_n_priors(keys, x, mask_type):
    model_fn = partial(net, state.params_ema)
    return sample_prior(keys, x, model_fn, mask_type)

@partial(jax.vmap, in_axes=(0, None, None, None, None))
def sample_n_conditionals(key, x_test, x_context, y_context, model_fn):
    return process.batch_conditional_sample(
        key, x_test, mask=MASK_TYPE_NOMASK, x_context=x_context, y_context=y_context, model_fn=model_fn)

# prior and process plots
def process_plots(batch, state, samplingkey) -> plt.figure:
    ts = (500, 300, 200, 100, 50, 25, 1)
    f,s,n = sample_prior(samplingkey, batch.x_target, mask_type=MASK_TYPE_NOMASK, model_fn=partial(net, state.params_ema))
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(num_channels, len(ts), figsize=(12,6))
    for i in range(num_channels):
        for j,t in enumerate(ts):
            ax[i,j].scatter(batch.x_target[i], s[500-t,i], s=2,alpha=0.6)
            ax[i,j].scatter(batch.x_target[i], n[500-t,i], s=2,alpha=0.6)
            ax[i,j].set_title(f'C={i}, t={t}')
    fig.suptitle('Denoised image through time')
    fig.tight_layout()
    fig.legend(['Denoised Prediction', 'Noise prediction'])

    return fig

def conditional_plots(batch, state, samplingkey) -> plt.figure:
    num_ctx = 10

    x_context = batch.x_target[:,:num_ctx,:]
    y_context = batch.y_target[:,:num_ctx,:]
    x_target = batch.x_target[:, num_ctx:, :]
    y_target = batch.y_target[:, num_ctx:, :]

    sampkeys = jax.random.split(samplingkey, 8)
    n_condsamps = sample_n_conditionals(sampkeys, x_target, x_context, y_context, partial(net, state.params_ema))
    mean, var = jnp.mean(n_condsamps, axis=0).squeeze(axis=-1), jnp.var(n_condsamps, axis=0).squeeze(axis=-1)
    fig,ax = plt.subplots(1,num_channels, figsize=(12,8))
    for i in range(num_channels):
        args = x_target[i].squeeze(-1).argsort()
        # ax[i].plot(x_target[i].squeeze(-1)[args], condsamps[i].squeeze(-1)[args])
        ax[i].plot(x_target[i].squeeze(-1)[args], mean[i][args])
        ax[i].fill_between(
            x_target[i].squeeze(-1)[args],
            mean[i][args] - 1.96*jnp.sqrt(var[i][args]),
            mean[i][args] + 1.96*jnp.sqrt(var[i][args]),
            color="k",
            alpha=0.1
        )
        ax[i].scatter(x_context[i], y_context[i], c='red', label="context points")
        ax[i].scatter(x_target[i], y_target[i], c='black', label="context points", s=6, marker="x")
        ax[i].set_title(f"Output {i}")
    fig.suptitle(f"Conditional samples from model when $step={state.step}$\nred=context, black=target, 10 context points.")
    fig.tight_layout()

    return fig

def create_plots(state, key, batch=None) -> None:
    # Create output directory if it doesn't exist
    output_dir = SAVE_HERE/'plots'
    os.makedirs(output_dir, exist_ok=True)
    bkey, k1,k2 = jax.random.split(key, 3)

    # Generate a testbatch
    if batch is None:
        batch = make_batch(bkey, 1, kernels['se'], coreg_weights, 1, -1, 100)
    else:
        batch = batch

    # Call process_plots and save the figure
    process_fig = process_plots(batch, state, k1)
    process_fig.savefig(output_dir/f"process_plot_step_{state.step}.png")
    plt.close(process_fig)

    # Call conditional_plots and save the figure
    conditional_fig = conditional_plots(batch, state, k2)
    conditional_fig.savefig(output_dir/f"conditional_plot_step_{state.step}.png")
    plt.close(conditional_fig)

# load pretrained checkpoint
# pretrained_state: TrainingState = init_state_from_pickle(os.path.abspath(PRETRAINED_MODEL_DIR))

# def traverse_and_switch(*, to_place, place_in):
#     for m, n, v in traverse(place_in):
#         pattern = (re.search(r'(?:\w+/)*\bbi_dimensional_attention_block[_\d+]*\b(?:\/\w+)*', m))
#         if bool(pattern): #if module name in new params contains bi_dimensional_attention_block
#             name = re.sub(r'^(?:\w+\/)*\bbi_dimensional_attention_block', 'bi_dimensional_attention_model/bi_dimensional_attention_block', m) #new params will be
#             v = to_place[name][n]
#         else:
#             v = v
#         # place back in
#         place_in[m][n] = v
#     # manually switch the linear layers
#     # linear, linear_1, linear_2
#     place_in['multi_channel_bdam/linear'] = to_place['bi_dimensional_attention_model/linear']
#     place_in['multi_channel_bdam/linear_1'] = to_place['bi_dimensional_attention_model/linear_1']
#     place_in['multi_channel_bdam/linear_2'] = to_place['bi_dimensional_attention_model/linear_2']

@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask_type=MASK_TYPE_NOMASK
    )
    #traverse_and_switch(to_place=pretrained_state.params_ema, place_in=initial_params)
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
    x_target=jnp.zeros((config.training.batch_size * num_channels, config.dataset.sample_length, INPUT_DIM)),
    y_target=jnp.zeros((config.training.batch_size * num_channels, config.dataset.sample_length, 1)),
)
state: TrainingState = init(batch_init, jax.random.PRNGKey(config.seed))

aimwriter = AimWriter(EXPERIMENT)
cfg_dict = asdict(config)
cfg_dict['mask_type'] = 'no mask'
cfg_dict['coreg_weights'] = coreg_weights.tolist()
aimwriter.log_hparams(cfg_dict)


actions = [
    actions.PeriodicCallback(
        every_steps=steps_per_epoch//8,
        callback_fn=lambda step, t, **kwargs: aimwriter.write_scalars(step, kwargs["metrics"])
    ),
    actions.PeriodicCallback(
        every_steps=10*steps_per_epoch,
        callback_fn=lambda step, t, **kwargs: state_utils.save_checkpoint(kwargs["state"], SAVE_HERE, step)
    ),
    actions.PeriodicCallback(
        every_steps=10*steps_per_epoch,
        callback_fn=lambda step, t, **kwargs: create_plots(kwargs['state'], kwargs['key'], kwargs["plotbatch"])
    ),
    actions.PeriodicCallback(
        every_steps=steps_per_epoch,
        callback_fn=lambda step, t, **kwargs: kwargs['writer'].writerow(
            [
                step,
                kwargs['metrics']['loss'],
                kwargs['metrics']['lr'],
                loss_fn(params=kwargs["state"].params_ema, batch=kwargs["valbatch"], key=kwargs["key"])
            ]
        )
    )
]
dkey = jax.random.key(53)
plotbatch = make_batch(dkey, 1, kernels["se"], coreg_weights, 1, -1, 100)
valbatch = make_batch(dkey, 32, kernels["se"], coreg_weights, 1, -1, 100)
steps = range(state.step + 1, NUM_STEPS + 1)
progress_bar = tqdm.tqdm(steps)


with open(SAVE_HERE/'metrics.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['step', 'loss', 'learning_rate', 'val_loss'])

    for step, batch in zip(progress_bar, ds_train):
        if step < state.step: continue  # wait for the state to catch up in case of restarts
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(state.step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=Params(state.params, state.params_ema, step), key=state.key, batch=batch, writer=csvwriter, plotbatch=plotbatch, valbatch=valbatch)

        if step % 32 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.3f}")

print(f'============================ {EXPERIMENT} finished training ===============================')
dict_to_yaml(cfg_dict, SAVE_HERE/'config.yaml')
with open(SAVE_HERE/'latest_trainingstate.pkl', 'wb') as file:
    pickle.dump(Params(state.params, state.params_ema, state.step), file)
