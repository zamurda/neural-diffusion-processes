import jax
import jax.numpy as jnp
from ml_tools.state_utils import load_checkpoint, TrainingState
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt
import aim
import io
from PIL import Image
from datetime import datetime
import equinox as eqx
import sys
import pathlib

from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.types import Batch, Rng
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion, EpsModel

from util.config_tools import get_config_map, parse_config_map, yaml_to_dict, Config, DatasetConfig
from util.data import gen_batch_eval
from functools import partial

import os
os.environ['JAX_PLATFORMS'] = 'cpu'


checkpoint = './trained_models/wind_Aug04_xfjd_testrun'
step_index = 4000

config_map = yaml_to_dict(pathlib.Path(checkpoint)/'config.yaml')
config: Config = parse_config_map(config_map)

# config_map = get_config_map()
# config: Config = parse_config_map(config_map)
key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)

MASK_TYPE_CAUSAL = jnp.array([])
MASK_TYPE_NOMASK = jnp.array([[]])

@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask_type):
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads
    )
    return model(x, y, t, mask_type)
    # return partial(model, mask_type=mask_type)(x, y, t)

# network_with_mask = hk.without_apply_rng(hk.transform(partial(network, mask_type=jnp.array([]))))


def net(params, t, yt, x, mask_type, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs
    # return network.apply(params, t[None], yt[None], x[None], mask_type)[0]
    return network.apply(params, t[None], yt[None], x[None], mask_type)[0]


steps_per_epoch = 4000 // config.training.num_epochs
learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=config.optimizer.init_lr,
    peak_value=config.optimizer.peak_lr,
    warmup_steps=steps_per_epoch * config.optimizer.num_warmup_epochs,
    decay_steps=steps_per_epoch * config.optimizer.num_decay_epochs,
    end_value=config.optimizer.end_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)


def init(batch: Batch, key) -> TrainingState:
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
INPUT_DIM = 1
batch_init = Batch(
    x_target=jnp.zeros((config.training.batch_size, config.dataset.sample_length, INPUT_DIM)),
    y_target=jnp.zeros((config.training.batch_size, config.dataset.sample_length, 1)),
)

state = init(batch_init, jax.random.PRNGKey(53))
loaded_state = load_checkpoint(state, checkpoint, step_index)
net_with_params = partial(net, loaded_state.params)

def sample_prior(key: Rng, x_target, model_fn):
    y0 = process.sample(key, x_target, mask=None, model_fn=model_fn)
    return y0

@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def sample_n_priors(key, x_target):
    return sample_prior(key, x_target, net_with_params)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def sample_n_conditionals(key, x_test, x_context, y_context, mask_context):
    return process.conditional_sample(
        key, x_test, mask=None, x_context=x_context, y_context=y_context, mask_context=mask_context, model_fn=net_with_params)


def plot_priors(batch: Batch, index_pos=0, timestamp=False):
    batch_size = len(batch.x_target)
    n = int(batch_size ** 0.5)
    # fig, axes = plt.subplots(n,n,figsize=(5,5), sharey=True, sharex=True)
    # axes = np.array(axes).reshape(-1)
    figs = [ ]
    for i in range(batch_size):
        fig, axes = plt.subplots(1,2, sharex=True, figsize=(12,8))

        samples = sample_n_priors(
            jax.random.split(jax.random.PRNGKey(42), 8),
            batch.x_target[i, :, index_pos, None],
        )
        mean, var = jnp.mean(samples, axis=0).squeeze(axis=1), jnp.var(samples, axis=0).squeeze(axis=1)
        axes[0].plot(batch.x_target[i, :, index_pos], samples[..., 0].T, "C0", lw=1)
        axes[0].plot(batch.x_target[i, :, index_pos], mean, "k")
        axes[0].fill_between(
            batch.x_target[i, :, index_pos, None].squeeze(axis=1),
            mean - 1.96 * jnp.sqrt(var),
            mean + 1.96 * jnp.sqrt(var),
            color="k",
            alpha=0.1,
        )
        axes[1].plot(batch.x_target[i, :, index_pos], batch.y_target[i], 'C0', lw=1)
        if timestamp:
            timestamps = batch.x_target[i, :, index_pos].tolist()
            dates = [ ]
            for t in timestamps:
                dates.append(datetime.fromtimestamp(t).strftime('%m-%d %H:%M:%S'))
            axes[0].set_xticks(timestamps, labels=dates, rotation=45, fontsize=4)
            axes[1].set_xticks(timestamps, labels=dates, rotation=45, fontsize=4)
        figs.append(fig)

    return figs

def plot_conditionals(batch: Batch, index_pos=0, timestamp=False):
    batch_size = len(batch.x_context)
    n = int(batch_size ** 0.5)
    fig, axes = plt.subplots(2,batch_size//2, figsize=(12,8), sharex=True, sharey=True)
    fig.suptitle(f'Power vs Wind speed, step {step_index}')
    axes = np.ravel(axes)
    metrics = {'mse': [], 'coverage_prob': []}
    means = []
    for i in range(batch_size):
        yc = (batch.y_context[i] - config_map['target_transformation']['mean']) / config_map['target_transformation']['std'] #scale context down
        samples = sample_n_conditionals(
            jax.random.split(jax.random.PRNGKey(42), 10),
            batch.x_target[i],
            batch.x_context[i],
            yc,
            None
        )
        samples = (samples * config_map['target_transformation']['std']) + config_map['target_transformation']['mean'] #rescale samples up
        y = evalbatch.y_target[i,:].squeeze(-1)
        mean, var = jnp.mean(samples, axis=0).squeeze(axis=1), jnp.var(samples, axis=0).squeeze(axis=1)
        # means.append(mean)
        coverage = 100*(jnp.count_nonzero((y >= mean - 1.96*jnp.sqrt(var)) & (y <= mean + 1.96*jnp.sqrt(var)))/len(y))
        mae = jnp.mean(jnp.abs(y-mean))
        metrics['mse'].append(mae)
        metrics['coverage_prob'].append(coverage)

        args = batch.x_target[i, :, index_pos].argsort()
        mean, var = mean[args], var[args]
        axes[i].plot(batch.x_target[i, :, index_pos][args], mean, "k")
        axes[i].fill_between(
            batch.x_target[i, :, index_pos, None].squeeze(axis=1)[args],
            mean - 1.96 * jnp.sqrt(var),
            mean + 1.96 * jnp.sqrt(var),
            color="k",
            alpha=0.1,
        )
        xc = batch.x_context[i, :, index_pos][:, None]
        axes[i].plot(xc, batch.y_context[i], "C3o")
        axes[i].set_title(f'MAE: {jnp.round(mae,3)} \n CVG(%): {coverage}')
        # axes[i].set_xlim(-2.05, 2.05)
        axes[i].scatter(batch.x_target[i, :, index_pos], batch.y_target[i].squeeze(axis=1), color='blue', marker='x')
        if timestamp:
            timestamps = jnp.concat((batch.x_context[i, :, index_pos], batch.x_target[i, :, index_pos])).tolist()
            dates = [ ]
            for t in timestamps:
                dates.append(datetime.fromtimestamp(t).strftime('%m-%d %H:%M:%S'))
            axes[i].set_xticks(timestamps, labels=dates, rotation=45, fontsize=4)
    # jnp.save('means', jnp.stack(means))
    # jnp.save('y', batch.y_target)
    return fig, metrics

dsc = DatasetConfig(
    'wind_1wk_turbine1_eval_is.pkl',
    features = ['Wind.speed.me'],
    target_index = 'Power.me',
    sample_length = 100
)                                     
n_draws = 6
evalbatch = gen_batch_eval(
    './data',
    dsc,
    n_draws=n_draws,
    make_context=True,
    target_is_normalised=True,
    num_context_points=20
) # we will normalise the context and then rescale

cond_plot, metrics = plot_conditionals(evalbatch)
print(metrics)
plt.show()

#############################################################################################################################################################################################
# Wq = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear']['w']
# bq = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear']['b']

# Wk = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear_1']['w']
# bk = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear_1']['b']

# Wv = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear_2']['w']
# bv = loaded_state.params['bi_dimensional_attention_model/bi_dimensional_attention_block/multi_head_attention/linear_2']['b']

# w_embed = loaded_state.params['bi_dimensional_attention_model/linear']['w']
# b_embed = loaded_state.params['bi_dimensional_attention_model/linear']['b']

# # data = jnp.ones(shape=(2,10,2,64))
# jnp.set_printoptions(threshold=sys.maxsize, linewidth=200)
# jax.config.update('jax_enable_x64', True)
# dsc = DatasetConfig(
#     'wind_1wk_turbine1_eval_is.pkl',
#     features = ['Wind.speed.me'],
#     target_index = 'Power.me',
#     sample_length = 10
# )
# _, batch = gen_batch_eval('./data', dsc, n_draws=3)
# x = batch.x_target
# y = batch.y_target
# num_x_dims = jnp.shape(x)[-1]
# x = jnp.expand_dims(x, axis=-1)
# y = jnp.repeat(jnp.expand_dims(y, axis=-1), num_x_dims, axis=2)
# data = jnp.concatenate([x, y], axis=-1)
# print(data.shape)

# data = jnp.matmul(data, w_embed) + b_embed
# data = jnp.swapaxes(data, 1, 2)
# print(data.shape)

# # sys.exit()
# num_heads = 8
# d_model = Wq.shape[1]
# depth = d_model//num_heads

# q = jnp.matmul(data, Wq) + bq
# k = jnp.matmul(data, Wk) + bk
# v = jnp.matmul(data, Wv) + bv
# print(q.shape, k.shape, v.shape)

# from einops import rearrange
# rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
# q = rearrange(q, rearrange_arg, num_heads=num_heads, depth=depth)
# k = rearrange(k, rearrange_arg, num_heads=num_heads, depth=depth)
# v = rearrange(v, rearrange_arg, num_heads=num_heads, depth=depth)
# print(q.shape, k.shape, v.shape)

# seq_len, batch_size = q.shape[-2], q.shape[0]
# mask = jnp.stack(
#     [jnp.tril(jnp.ones((seq_len, seq_len)), k=-1)] * batch_size
#     )
# mask = mask[...,None,None,:,:]
# print(mask.shape)
# # print(mask[0,0,0,...])
# # sys.exit()
# qk = jnp.einsum('...qd,...kd -> ...qk', q, k)
# print(qk.shape)
# print('\nattn pattern')
# print(qk[0,0,0,...]) # first attn pattern

# depth = jnp.shape(k)[-1] * 1.0
# scaled_attention_logits = qk / jnp.sqrt(depth)
# print('\n scaled logits')
# print(scaled_attention_logits[0,0,0,...])
# scaled_attention_logits += mask * -1e9
# print('\n scaled logits + mask')
# print(scaled_attention_logits[0,0,0,...])
# attention_weights = jax.nn.softmax(
#         scaled_attention_logits, axis=-1
#     )
# print('\n attn weights')
# print((attention_weights[0,0,0,...])) # first attn pattern again
