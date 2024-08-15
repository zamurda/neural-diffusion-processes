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
import gpflow

from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.types import Batch, Rng
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion, EpsModel

from util.config_tools import get_config_map, parse_config_map, Config, DatasetConfig
from util.data import gen_batch_eval
from functools import partial

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

config_map = get_config_map()
config: Config = parse_config_map(config_map)


checkpoint = './trained_models/gp_Aug01_tyjj_testrun'
step_index = 19200

config_map = get_config_map()
config: Config = parse_config_map(config_map)
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


def net(params, t, yt, x, mask_type, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs

    return network.apply(params, t[None], yt[None], x[None], mask_type)[0]

NUM_STEPS = (4096 // 32) * config.training.num_epochs
steps_per_epoch = NUM_STEPS // config.training.num_epochs
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
net_with_params = partial(net, loaded_state.params_ema)

def sample_prior(key: Rng, x_target, model_fn):
    y0 = process.sample(key, x_target, mask=MASK_TYPE_NOMASK, model_fn=model_fn)
    return y0

@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def sample_n_priors(key, x_target):
    return sample_prior(key, x_target, net_with_params)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def sample_n_conditionals(key, x_test, x_context, y_context, mask_context):
    return process.conditional_sample(
        key, x_test, mask=MASK_TYPE_NOMASK, x_context=x_context, y_context=y_context, mask_context=mask_context, model_fn=net_with_params)

key = jax.random.key(1)
x = jax.random.uniform(key , (100,1), minval=-1, maxval=1)

K = gpflow.kernels.SquaredExponential(1,0.25)(x)
func = np.random.multivariate_normal([0]*100, K).reshape(100,1)


ncond = (1,2,4,8,16,32,64)
# jnp.save('x_test', x)
# jnp.save('y_test', func)
args = x.argsort(axis=0)
fig,ax = plt.subplots(1,len(ncond)+1, sharex=True, sharey=True)
ax[0].plot(x.squeeze(-1)[args], func.squeeze(-1)[args])
for i,n in enumerate(ncond):
    key, k1, k2 = jax.random.split(key, 3)
    context_pos = jax.random.choice(k1, jnp.arange(100), (n,), replace=False)
    x_context = x[context_pos]
    y_context = func[context_pos]
    target = jnp.delete(x, context_pos)[:,None]
    t_args = target.argsort(axis=0)
    
    # jnp.save(f'context_pos_{i}', context_pos)
    keys = jax.random.split(k2, 8)
    samples = sample_n_conditionals(keys, target, x_context, y_context, None)
    for j in range(samples.shape[0]):
        ax[i+1].plot(target.squeeze(-1)[t_args], samples[j].squeeze(-1)[t_args], lw=1, c='blue', alpha=0.5)
        ax[i+1].scatter(x_context, y_context, marker='x', c='red')
    # jnp.save(f'cond_samples_{i}', samples)
    print(f'processed cond samples {i}')
plt.show()
    
# run = aim.Run('bf1ab5c00490426183db0b77')
# for i, (fig1, fig2) in enumerate(zip(prior_plots, cond_plots_random_ctx)):
#     fig_map = {
#         f'prior_draw_{i}': fig1,
#         f'cond_random_draw_{i}': fig2,
#         }
#     for name, fig in fig_map.items():
#         img_buf = io.BytesIO()
#         fig.savefig(img_buf, format="png")
#         im = Image.open(img_buf)
#         im = aim.Image(im)
#         plt.close(fig)
#         run.track(value=im, name=name)
