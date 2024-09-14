from typing import Mapping, Tuple, List, Union
from jaxtyping import Array, PyTree
import os
import sys
import pprint
import string
import random
import datetime
import pathlib
import jax
import haiku as hk
import jax.numpy as jnp
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import optax
import equinox as eqx
from functools import partial
from dataclasses import asdict
import csv
import pickle

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

from util.config_tools import get_config_map, parse_config_map, Config, DatasetConfig, dict_to_yaml
from util.data import gen_dataset_multi

class Params(eqx.Module):
    params: PyTree
    params_ema: PyTree
    step: int

# train on apple GPU if told to
flags.FLAGS(sys.argv)
if flags.FLAGS.metal:
    print('========================= using metal GPU ============================')
    os.environ['JAX_PLATFORMS'] = 'gpu'
else:
    print('========================= using CPU ============================')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    jax.config.update('jax_enable_x64', True) # enable double precision

# jax.config.update('jax_disable_jit', True) #disable jit for debugging
# jax.config.update("jax_debug_nans", True) #nan debugging mode
# experiment metadata
timestamp = datetime.datetime.now().strftime("%b%d")
letters = string.ascii_lowercase
id = ''.join(random.choice(letters) for i in range(4))
EXPERIMENTS_DIR = './trained_models'
EXPERIMENT = f'multi_wind_{timestamp}_{id}'
DATA_DIR = pathlib.Path('./data')
# LOG_DIR = pathlib.Path('./logs')

SAVE_HERE = pathlib.Path(EXPERIMENTS_DIR)/pathlib.Path(EXPERIMENT)
if not SAVE_HERE.exists():
    os.mkdir(SAVE_HERE)

MASK_TYPE_CAUSAL = jnp.array([])
MASK_TYPE_NOMASK = jnp.array([[]])


config_map: Mapping = get_config_map()
config: Config = parse_config_map(config_map)
key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)

num_channels = len(config.dataset.data)
ds_train, NUM_STEPS, meanvar_dict = gen_dataset_multi(
                                        DATA_DIR,
                                        config.dataset,
                                        config.training,
                                        config.seed,
                                        target_is_normalised=False
)

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
        n_channels=num_channels
        )
    return ndp.process.loss_multichannel(process, net_with_params, batch, key, **kwargs)

steps_per_epoch = NUM_STEPS // config.training.num_epochs
# learning_rate_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=config.optimizer.init_lr,
#     peak_value=config.optimizer.peak_lr,
#     warmup_steps=steps_per_epoch * config.optimizer.num_warmup_epochs,
#     decay_steps=steps_per_epoch * config.optimizer.num_decay_epochs,
#     end_value=config.optimizer.end_lr,
# )
learning_rate_schedule = optax.constant_schedule(3e-4)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask_type=MASK_TYPE_NOMASK
    )
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )


@jax.jit
def ema_update(decay, ema_params, new_params):
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params
    return jax.tree_map(_ema, ema_params, new_params)


@jax.jit
def update_step(state: TrainingState, batch: Batch) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
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

INPUT_DIM = len(config.dataset.features)
batch_init = Batch(
    x_target=jnp.zeros((config.training.batch_size * num_channels, config.dataset.sample_length, INPUT_DIM)),
    y_target=jnp.zeros((config.training.batch_size * num_channels, config.dataset.sample_length, 1)),
)
state: TrainingState = init(batch_init, jax.random.PRNGKey(config.seed))

# sys.exit()
#writer = AimWriter(EXPERIMENT)
cfg_dict = asdict(config)
cfg_dict['mask_type'] = 'no mask'
cfg_dict['target_transformation'] = meanvar_dict
# writer.log_hparams(cfg_dict)

# net_with_params = partial(net, state.params_ema)

def sample_prior(key: Rng, x_target, model_fn):
    y0 = process.sample(key, x_target, mask=MASK_TYPE_NOMASK, model_fn=model_fn)
    return y0

@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def sample_n_priors(key, x_target, state):
    net_with_params = partial(net, state.params_ema)
    return sample_prior(key, x_target, net_with_params)

def prior_plots(batch: Batch, state: TrainingState) -> plt.figure:
    '''meant to plot prior samples periodically. just do the first batch_size//4'''
    fig, axes = plt.subplots(config.training.batch_size//4,2, figsize=(12,36), sharex=True, sharey=True)
    # net_with_params = partial(net, state.params_ema)
    for i in range(config.training.batch_size//4):
        args = batch.x_target[i, :, 0].argsort()
        samples = sample_n_priors(
            jax.random.split(jax.random.PRNGKey(42), 8),
            batch.x_target[i, :, 0, None],
            state
        )
        mean, var = jnp.mean(samples, axis=0).squeeze(axis=1)[args], jnp.var(samples, axis=0).squeeze(axis=1)[args]
        axes[i,0].plot(batch.x_target[i, :, 0][args], samples[..., 0].T[args], "-o", lw=1, markersize=3)
        axes[i,0].plot(batch.x_target[i, :, 0][args], mean, "k")
        axes[i,0].fill_between(
            batch.x_target[i, :, 0, None].squeeze(axis=1)[args],
            mean - 1.96 * jnp.sqrt(var),
            mean + 1.96 * jnp.sqrt(var),
            color="k",
            alpha=0.1,
        )
        axes[i,1].plot(batch.x_target[i,:,0][args], batch.y_target[i,:,0][args], "-o", lw=1, markersize=3)
        # axes[1].scatter(batch.x_target[i, :, 0], batch.y_target[i], 'C0', lw=1)

    return fig

def process_plots(batch: Batch, state: TrainingState) -> plt.figure:
    '''plots predicted noise at predetermined timesteps for a random function sample'''
    key, subkey = jax.random.split(jax.random.key(config.seed))
    net_with_params = partial(net, state.params_ema, mask_type=MASK_TYPE_NOMASK)
    
    chosen_idx = jax.random.choice(subkey, jnp.arange(batch.x_target.shape[0]))
    x = batch.x_target[chosen_idx]
    y = batch.y_target[chosen_idx]
    args = x.squeeze(-1).argsort()
    ts = (25,50,100,200,300,400,500)

    fig,ax = plt.subplots(3,len(ts)+1, sharex=True, figsize=(16,10))
    ax[0,0].plot(x.squeeze(-1)[args], y.squeeze(-1)[args])
    ax[0,0].set_title('original')
    for i,t in enumerate(ts):
        key, subkey = jax.random.split(key)
        yt, noise = process.forward(subkey, y, t)
        noise_hat = net_with_params(jnp.array([t]).squeeze(-1), yt, x, key=subkey)
        loss = jnp.round(jnp.array([jnp.sum(jnp.sum(jnp.abs(noise - noise_hat), axis=1))/100]), 3)

        ax[0,i+1].plot(x.squeeze(-1)[args], yt.squeeze(-1)[args])
        ax[0,i+1].set_title(f't = {t}')
        
        ax[1,i+1].scatter(x.squeeze(-1), noise.squeeze(-1))
        ax[1,i+1].set_title('true noise')

        ax[2,i+1].scatter(x.squeeze(-1), noise_hat.squeeze(-1))
        ax[2,i+1].set_title(f'pred noise \n L = {loss}')
    return fig

actions = [
    actions.PeriodicCallback(
        every_steps=NUM_STEPS//config.training.num_epochs,
        callback_fn=lambda step, t, **kwargs: state_utils.save_checkpoint(kwargs["state"], SAVE_HERE, step)
    ),
    actions.PeriodicCallback(
        every_steps=steps_per_epoch,
        callback_fn=lambda step, t, **kwargs: kwargs['writer'].writerow(
            [
                step,
                kwargs['metrics']['loss'],
                kwargs['metrics']['lr'],
            ]
        )
    )
    #actions.PeriodicCallback(
    #    every_steps=4*steps_per_epoch,
    #    callback_fn=lambda step, t, **kwargs:writer.write_figures(
    #        step,
    #        {
    #            'prior samples':prior_plots(batch=kwargs['batch'], state=kwargs['state']),
    #            'predicted noise':process_plots(batch=kwargs['batch'], state=kwargs['state'])
    #        }
    #    )
    #),
]

print(f"Starting experiment {EXPERIMENT}")
steps = range(state.step + 1, state.step + NUM_STEPS + 1)
progress_bar = tqdm.tqdm(steps)

with open(SAVE_HERE/'metrics.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    if state.step <= 1:
        csvwriter.writerow(['step', 'loss', 'learning_rate'])

    for step, batch in zip(progress_bar, ds_train):
        if step < state.step: continue  # wait for the state to catch up in case of restarts
        if step > 3: break
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(state.step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=Params(state.params, state.params_ema, step), key=state.key, batch=batch, writer=csvwriter)

        if step % 32 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.3f}")


print(f'===================== EXPERIMENT {EXPERIMENT} DONE ===================')
dict_to_yaml(cfg_dict, SAVE_HERE/'config.yaml')
with open(SAVE_HERE/'latest_trainingstate.pkl', 'wb') as file:
    pickle.dump(Params(state.params, state.params_ema, state.step), file)