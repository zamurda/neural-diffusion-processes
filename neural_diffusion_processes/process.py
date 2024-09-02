from typing import Protocol, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from einops import rearrange

from .types import Batch, Rng, ndarray


class EpsModel(Protocol):
    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "mask_type: [...]", "return: [N, y_dim]")
    def __call__(self, t: ndarray, yt: ndarray, x: ndarray, mask_type: ndarray, *, key: Rng) -> ndarray:
        ...


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def cosine_schedule(beta_start, beta_end, timesteps, s=0.008, **kwargs):
    x = jnp.linspace(0, timesteps, timesteps + 1)
    ft = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0.0001, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class GaussianDiffusion:
    betas: ndarray
    alphas: ndarray
    alpha_bars: ndarray

    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = jnp.cumprod(1.0 - betas)

    @check_shapes("y0: [N, y_dim...]", "t: []", "return[0]: [N, y_dim...]", "return[1]: [N, y_dim...]")
    def pt0(self, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        alpha_bars = expand_to(self.alpha_bars[t], y0)
        m_t0 = jnp.sqrt(alpha_bars) * y0
        v_t0 = (1.0 - alpha_bars) * jnp.ones_like(y0)
        return m_t0, v_t0

    @check_shapes("y0: [N, y_dim...]", "t: []", "return[0]: [N, y_dim...]", "return[1]: [N, y_dim...]")
    def forward(self, key: Rng, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        m_t0, v_t0 = self.pt0(y0, t)
        noise = jax.random.normal(key, y0.shape)
        yt = m_t0 + jnp.sqrt(v_t0) * noise
        return yt, noise
    
    @check_shapes("y0: [N, y_dim...]", "t: []", "return[0]: [N, y_dim...]", "return[1]: [N, y_dim...]")
    def fwd_with_same_noise(self, key: Rng, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        m_t0, v_t0 = self.pt0(y0, t)
        noise = jax.random.normal(key, (y0.shape[1])) # sample noise of size seq_len so that the same noise vector can be applied over all input samples
        fullnoise=  jnp.tile(noise, (y0.shape[0], 1))[...,None]
        yt = m_t0 + jnp.sqrt(v_t0) * fullnoise
        return yt, fullnoise

    def ddpm_backward_step(self, key: Rng, noise: ndarray, yt: ndarray, t: ndarray) -> ndarray:
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)
        t = t[0] if t.size > 0 and t.ndim > 0 else t
        z = (t > 0) * jax.random.normal(key, shape=yt.shape, dtype=yt.dtype)

        a = 1.0 / jnp.sqrt(alpha_t)
        b = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        yt_minus_one = a * (yt - b * noise) + jnp.sqrt(beta_t) * z
        return yt_minus_one

    def ddpm_backward_mean_var(self, noise: ndarray, yt: ndarray, t: ndarray) -> ndarray:
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        a = 1.0 / jnp.sqrt(alpha_t)
        b = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        m = a * (yt - b * noise)
        v = beta_t * jnp.ones_like(yt) * (t > 0)
        v = jnp.maximum(v, jnp.ones_like(v) * 1e-3)
        return m, v

    def sample(self, key, x, mask, *, model_fn: EpsModel, batched_input: bool = False, output_dim: int = 1):
        """returns the noise at each intermediate step too"""
        
        key, ykey = jax.random.split(key)
        
        if batched_input:
            shape = list(x.shape[:-1]) + [1] # make sure last dim of x is 1
            yT = jax.random.normal(ykey, shape)
            #yT = jnp.tile(jax.random.normal(ykey, shape[1:]), (x.shape[0], 1, 1)) #same starting noise
            ts = jnp.tile(jnp.arange(len(self.betas))[::-1], (x.shape[0], 1)).T #[B, num_timesteps]
        else:
            yT = jax.random.normal(ykey, (len(x), output_dim))
            ts = jnp.arange(len(self.betas))[::-1]
        
        @jax.jit
        def scan_fn(y, inputs):
            t, key = inputs
            mkey, rkey = jax.random.split(key)
            noise_hat = model_fn(t, y, x, mask, key=mkey)
            y = self.ddpm_backward_step(key=rkey, noise=noise_hat, yt=y, t=t)
            return y, (y, noise_hat)

        keys = jax.random.split(key, len(ts))
        yf, yt = jax.lax.scan(scan_fn, yT, (ts, keys))
        return yf, yt[0], yt[1]

    def conditional_sample(
        self,
        key,
        x,
        mask,
        *,
        x_context,
        y_context,
        mask_context,
        model_fn: EpsModel,
        num_inner_steps: int = 5,
        method: str = "repaint",
    ):
        if mask is None:
            mask = jnp.zeros_like(x[:, 0])

        if mask_context is None:
            mask_context = jnp.zeros_like(x_context[:, 0])

        key, ykey = jax.random.split(key)
        x_augmented = jnp.concatenate([x_context, x], axis=0)
        mask_augmented = jnp.concatenate([mask_context, mask], axis=0)
        num_context = len(x_context)

        @jax.jit
        def repaint_inner(yt_target, inputs):
            t, key = inputs
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            # one step backward: t -> t-1
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, yt_target], axis=0)
            noise_hat = model_fn(t, y_augmented, x_augmented, mask_augmented, key=mkey)
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            # one step forward: t-1 -> t
            z = jax.random.normal(key, shape=y.shape)
            beta__t_minus_1 = expand_to(self.betas[t - 1], y)
            y = jnp.sqrt(1.0 - beta__t_minus_1) * y + jnp.sqrt(beta__t_minus_1) * z
            return y, None

        @jax.jit
        def repaint_outer(y, inputs):
            t, key = inputs
            # loop
            key, ikey = jax.random.split(key)
            ts = jnp.ones((num_inner_steps,), dtype=jnp.int32) * t
            keys = jax.random.split(ikey, num_inner_steps)
            y, _ = jax.lax.scan(repaint_inner, y, (ts, keys))

            # step backward: t -> t-1
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, y], axis=0)
            noise_hat = model_fn(t, y_augmented, x_augmented, mask_augmented, key=mkey)
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            return y, None

        ts = jnp.arange(len(self.betas))[::-1]
        keys = jax.random.split(key, len(ts))
        yT_target = jax.random.normal(ykey, (len(x), y_context.shape[-1]))

        y, _ = jax.lax.scan(repaint_outer, yT_target, (ts[:-1], keys[:-1]))
        return y
    
    def batch_conditional_sample(
    self,
    key,
    x,
    mask,
    *,
    x_context,
    y_context,
    model_fn: EpsModel,
    num_inner_steps: int = 5,
    method: str = "repaint",
):

        key, ykey = jax.random.split(key)
        x_augmented = jnp.concatenate([x_context, x], axis=1)
        mask_augmented = mask
        num_context = x_context.shape[1]
        num_channels = x.shape[0]

        @jax.jit
        def repaint_inner(yt_target, inputs):
            """
            inner loop - forward and backwards num_innner steps times.
            called via a scan over ts and keys of length num_inner
            """
            t, key = inputs
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            # one step backward: t -> t-1
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, yt_target], axis=1)
            noise_hat = model_fn(t.repeat(num_channels), y_augmented, x_augmented, mask_augmented, key=mkey)
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[:,num_context:]
            # one step forward: t-1 -> t
            z = jax.random.normal(key, shape=y.shape)
            beta__t_minus_1 = expand_to(self.betas[t - 1], y)
            y = jnp.sqrt(1.0 - beta__t_minus_1) * y + jnp.sqrt(beta__t_minus_1) * z
            return y, None

        @jax.jit
        def repaint_outer(y, inputs):
            t, key = inputs #[],[]
            # loop
            key, ikey = jax.random.split(key)
            ts = jnp.ones((num_inner_steps,), dtype=jnp.int32) * t
            # ts = jnp.tile(t, (num_inner_steps,1)).astype(jnp.int32)
            keys = jax.random.split(ikey, num_inner_steps)
            y, _ = jax.lax.scan(repaint_inner, y, (ts, keys)) #scan inner loop num_inner_steps times, taking singula keys and ts

            # step backward: t -> t-1
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, y], axis=1)
            noise_hat = model_fn(t.repeat(num_channels), y_augmented, x_augmented, mask_augmented, key=mkey) #expand t whenever inference is done
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[:, num_context:]
            return y, None

        ts = jnp.arange(len(self.betas))[::-1]
        keys = jax.random.split(key, len(ts))
        yT_target = jax.random.normal(ykey, x.shape)

        shape = list(x.shape[:-1]) + [1] # make sure last dim of x is 1
        yT_target = jax.random.normal(ykey, shape)

        y, _ = jax.lax.scan(repaint_outer, yT_target, (ts[:-1], keys[:-1]))
        return y


def loss(
    process: GaussianDiffusion,
    network: EpsModel,
    batch: Batch,
    key: Rng,
    mask_type: ndarray = jnp.array([[]]),
    *,
    num_timesteps: int,
    loss_type: str = "l1",
):
    if loss_type == "l1":

        def loss_metric(a, b):
            return jnp.abs(a - b)

    elif loss_type == "l2":

        def loss_metric(a, b):
            return (a - b) ** 2

    else:
        raise ValueError(f"Unknown loss type {loss_type}")

    @check_shapes(
        "t: []", "y: [N, y_dim]", "x: [N, x_dim]", "mask_type: [...]", "return: []"
    )
    def loss_fn(key, t, y, x, mask_type):
        yt, noise = process.forward(key, y, t)
        noise_hat = network(t, yt, x, mask_type, key=key)
        loss_value = jnp.sum(loss_metric(noise, noise_hat), axis=1)  # [N,]
        # loss_value = loss_value * (1.0 - mask)
        # num_points = len(mask) - jnp.count_nonzero(mask)
        num_points = y.shape[0]
        return jnp.sum(loss_value) / num_points

    batch_size = len(batch.x_target)

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=0, maxval=num_timesteps / batch_size)
    t = t + (num_timesteps / batch_size) * jnp.arange(batch_size)
    t = t.astype(jnp.int32)

    keys = jax.random.split(key, batch_size)

    # if batch.mask_target is None:
    #     # consider all points
    #     mask_target = jnp.zeros_like(batch.x_target[..., 0])
    # else:
    #     mask_target = batch.mask_target
    loss_with_mask = partial(loss_fn, mask_type=mask_type)

    losses = jax.vmap(loss_with_mask)(keys, t, batch.y_target, batch.x_target)
    return jnp.mean(losses)


def loss_multichannel(
    process: GaussianDiffusion,
    network: EpsModel,
    batch: Batch,
    key: Rng,
    *,
    mask_type: ndarray = jnp.array([[]]),
    n_channels: int,
    num_timesteps: int,
    loss_type: str = "l1",        
) -> jnp.ndarray:
    """ 
    loss for a multi-channel model. basically the same loss but t is the same within a channel
    """
    if loss_type == "l1":

        def loss_metric(a, b):
            return jnp.abs(a - b)

    elif loss_type == "l2":

        def loss_metric(a, b):
            return (a - b) ** 2

    else:
        raise ValueError(f"Unknown loss type {loss_type}")

    @check_shapes(
        "t: []", "y: [N, y_dim...]", "x: [N, x_dim...]", "mask_type: [...]", "return: []"
    )
    def loss_fn(key, t, y, x, mask_type):
        """
        in the multichannel case, the loss is first averaged over the channel dimensions
        each output in a channel is forwarded with the same noise (so that latent alignment can take place)
        """
        yt, noise = process.forward(key, y, t)
        t = t.repeat(y.shape[0])
        noise_hat = network(t, yt, x, mask_type, key=key)
        loss_value = jnp.mean(loss_metric(noise, noise_hat), axis=0).squeeze(-1)  # [N,]
        # loss_value = loss_value * (1.0 - mask)
        # num_points = len(mask) - jnp.count_nonzero(mask)
        num_points = y.shape[1]
        return jnp.sum(loss_value) / num_points

    batch_size = len(batch.x_target) // n_channels # (actual size of batch dimension)

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=0, maxval=num_timesteps / batch_size)
    t = t + (num_timesteps / batch_size) * jnp.arange(batch_size)
    t = t.astype(jnp.int32)

    keys = jax.random.split(key, batch_size)

    # rearrange so that one MOGP sample is processed at a time
    rearrange_arg = '(batch channel) seq_len ... -> batch channel seq_len ...'
    y = rearrange(batch.y_target, rearrange_arg, channel=n_channels)
    x = rearrange(batch.x_target, rearrange_arg, channel=n_channels)
    # t = rearrange(t, '(batch channel) -> batch channel', channel=n_channels) # then t will be scalar
    
    loss_with_mask = partial(loss_fn, mask_type=mask_type)
    losses = jax.vmap(loss_with_mask)(keys, t, y, x)
    return jnp.mean(losses)
