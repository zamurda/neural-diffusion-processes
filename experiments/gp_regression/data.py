import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import jax.numpy as jnp
import jax
import jax.dlpack
from einops import rearrange

from typing import Tuple, Callable
from functools import partial

from neural_diffusion_processes.types import Batch, Dataset, Rng

_SAMPLES_PER_EPOCH = 4096

matern = gpflow.kernels.Matern32(variance=1, lengthscales=0.25)
se = gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.25)

def tf_to_jax(arr):
    return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))

def jax_to_tf(arr):
    return tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(arr))

# @jax.jit(static_argnums=(1,2))
# def se_kernel(x, var, l):
#     assert(len(x.shape) == 2 and x.shape[-1] == 1)

#     K = jnp.ones(shape=(x.shape[0], x.shape[0]))
    # for i in range()


def make_batch(key, batch_size: int, sample_length:int, kern, xlow, xhigh) -> Batch:
    xkey, ykey, nkey = jax.random.split(key, 3)

    X = jax.random.uniform(xkey, shape=(batch_size,sample_length), minval=xlow, maxval=xhigh)
    Ks = [kern(X[i].reshape(sample_length,1)) for i in range(batch_size)]
    samples = np.stack([jax.random.multivariate_normal(ykey, jnp.array([0]*sample_length), tf_to_jax(Ks[i]), method='svd') for i in range(batch_size)])
    # X = rearrange(X[...,None], 'input_dim batch_size seq_len -> batch_size seq_len input_dim')
    del Ks # don't wait for GC
    return Batch(x_target=X[...,None], y_target=samples[...,None]) # are the shapes right?

def gen_dataset(num_epochs: int, **kwargs) -> Dataset:
    num_batches = num_epochs * (_SAMPLES_PER_EPOCH // kwargs['batch_size'])

    partial_make_batch = partial(
            make_batch,
            batch_size=kwargs['batch_size'],
            sample_length=kwargs['sample_length'],
            kern=kwargs['kernel'],
            xlow=kwargs['domain'][0],
            xhigh=kwargs['domain'][1]
        )
    key = jax.random.key(53)
    keys = jax.random.split(key, num_batches)

    return map(partial_make_batch, keys)


if __name__ == '__main__':

    # for testing

    dataset = gen_dataset(
        num_epochs=1,
        batch_size=32,
        sample_length=100,
        domain=(-1,1),
        kernel=se,
    )
    fig,ax = plt.subplots(1,1)
    for batch in dataset:
        print(batch.x_target.shape)
        for i in range(32):
            args = batch.x_target[i].argsort(axis=0)
            ax.plot(batch.x_target[i].squeeze(-1)[args], batch.y_target[i].squeeze(-1)[args], '-o', markersize=3)
    plt.show()
    

