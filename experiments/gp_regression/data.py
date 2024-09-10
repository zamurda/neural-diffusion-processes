import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import jax.dlpack
from einops import rearrange

from typing import Tuple, Callable
from functools import partial

from neural_diffusion_processes.types import Batch, Dataset, Rng

_SAMPLES_PER_EPOCH = 4096


def se_kernel(X: jnp.ndarray, sigma2: jnp.float32, l: jnp.float32) -> jnp.ndarray:
    """ 
    Computes the full squared-exponential kernel matrix for an input data matrix.
    """
    sq_dists = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = sigma2 * jnp.exp(-sq_dists / (2 * l ** 2))
    return K


def make_batch(key, batch_size: int, sample_length:int, kern, xlow, xhigh) -> Batch:
    xkey, ykey, nkey = jax.random.split(key, 3)
    samplekeys = jax.random.split(ykey,batch_size)
    X = jax.random.uniform(xkey, shape=(batch_size,sample_length,1), minval=xlow, maxval=xhigh) #[B,N,1]
    # K = kern(X)
    # Ks = [kern(X[i].reshape(sample_length,1)) for i in range(batch_size)]
    # samples = np.stack([jax.random.multivariate_normal(ykey, jnp.array([0]*sample_length), tf_to_jax(Ks[i]), method='svd') for i in range(batch_size)])
    samples = jax.vmap(lambda x,key: jax.random.multivariate_normal(key, jnp.array([0] * sample_length), kern(x), method="svd"), in_axes=(0,0))(X,samplekeys)
    return Batch(x_target=X, y_target=samples[...,None]) # are the shapes right?  

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

    return map(jax.jit(partial_make_batch), keys)


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
    

