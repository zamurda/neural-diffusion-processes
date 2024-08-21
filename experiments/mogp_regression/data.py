''' creates a multi-channel dataset sampled from an MOGP '''
from functools import partial
import jax
import jax.numpy as jnp
from einops import rearrange

from typing import Tuple, Callable
from jaxtyping import Float32, Array, PRNGKeyArray, Key

from neural_diffusion_processes.types import Batch, Dataset

_VAR = 1
_LENGTHSCALE = 0.25
_MOGP_SAMPLES_PER_EPOCH = 4096


def se_kernel(X: jnp.ndarray, sigma2: jnp.float32, l: jnp.float32) -> jnp.ndarray:
    """ 
    Computes the full squared-exponential kernel matrix for an input data matrix.
    """
    sq_dists = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = sigma2 * jnp.exp(-sq_dists / (2 * l ** 2))
    return K


# Only done this way because installed jax version (0.4.19) doesn't allow for reshaping key arrays.
@partial(jax.vmap, in_axes=(0,None,0,0))
def sample_heterotopic_mogp(key: Key, kernelfunc: Callable, x: Float32[Array, "N D"], coreg_weights: Float32[Array, "R"]) -> Float32[Array, "N"]:
    """
    Generates samples from a fully coregionalised MOGP according to the 'Intrinsic Model of Coregionalisation'
    To use: Pass as args a [C,N,D] array of data, a [C,] array of keys and a [C,R] array of mixing weights.
    """
    R = coreg_weights.shape[0]

    keys = (jax.random.split(key, R))
    sampler = jax.jit(lambda key, x: jax.random.multivariate_normal(key, jnp.array([0] * x.shape[0]), kernelfunc(x),  method='svd')) # can jit since key arg is known at call time
    independent_samples = jax.vmap(partial(sampler, x=x), in_axes=(0))(keys) #[R, N]
    
    return jnp.einsum('rn,r -> n', independent_samples, coreg_weights)

# @jax.jit
def make_batch(key, batch_size, kernelfunc, coreg_weights, maxval, minval, sample_length) -> Batch:
    """
    Generates a batch of size 'batch_size' * coreg_weights.shape[0]
    So function samples can be interpreted as batches and then rearranged to be
    processed as an instance of an MOGP.
    """
    key, subkey = jax.random.split(key)
    X = jax.random.uniform(subkey, (batch_size, coreg_weights.shape[0], sample_length, 1), minval=minval, maxval=maxval)
    keys = jax.random.split(key, batch_size)
    C = coreg_weights.shape[0]
    samples = jax.vmap(
        lambda key, x_b: sample_heterotopic_mogp(jax.random.split(key, C), kernelfunc, x_b, coreg_weights),
        in_axes=(0,0)
    )(keys, X)
    key, subkey = jax.random.split(subkey)
    samples += jax.random.normal(key, (samples.shape)) * 1e-3 # add observation noise

    rearrange_arg_x = "b c n d -> (b c) n d"
    rearrange_arg_y = "b c n -> (b c) n"
    return Batch(
        x_target=rearrange(X, rearrange_arg_x),
        y_target=rearrange(samples, rearrange_arg_y)[...,None]
    )

def gen_dataset(seed, kernelfunc, coreg_weights, x_minval, x_maxval, num_epochs, batch_size, sample_length) -> Dataset:
    """
    Returns generator over entire dataset
    """
    init_rng = jax.random.key(seed)
    mogp_samples_per_batch = batch_size
    batches_per_epoch = _MOGP_SAMPLES_PER_EPOCH // mogp_samples_per_batch
    num_steps = num_epochs * batches_per_epoch
    keys = jax.random.split(init_rng, num_steps)
    mb = jax.jit(partial(make_batch, batch_size=mogp_samples_per_batch, kernelfunc=kernelfunc, coreg_weights=coreg_weights, maxval=x_maxval, minval=x_minval, sample_length=sample_length))
    return map(mb, keys)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import gpflow
    from jax import dlpack
    import numpy as np
    import sys
    from functools import partial

    kernelfunc = jax.jit(partial(se_kernel, sigma2=1, l=0.25))
    seedkey = jax.random.key(53)
    k, sk = jax.random.split(seedkey)
    # X = jax.random.uniform(k, shape=(3,100,1), minval=-1, maxval=1) #[C,N,D]
    coreg_weights = jnp.array([
        [0.5, 0.5],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.1, 0.9]
    ]) # each gp made of two latent gps
    assert(jnp.min(jnp.linalg.eigvals(coreg_weights @ coreg_weights.T)) > 0)
    
    '''keys = jax.random.split(k, 4)
    X = jax.random.uniform(sk, shape=(4,3,100,1), minval=-1, maxval=1)
    batch = make_batch(keys, X, kernelfunc, coreg_weights)
    print(batch.x_target.shape)
    print(batch.y_target.shape)

    Y = rearrange(batch.y_target, '(b c) n 1 -> b c n 1', b=4, c=3)
    fig, ax = plt.subplots(4,3)
    for i in range(4):
        for j in range(3):
            args = X[i,j].argsort(axis=0)
            ax[i,j].plot(X[i,j][args].squeeze(-1), Y[i,j,...][args].squeeze(-1))
    plt.show()'''

    config = {
        'seed': 53,
        'kernelfunc': kernelfunc,
        'coreg_weights':coreg_weights,
        'x_minval':-1,
        'x_maxval':1,
        'num_epochs':1,
        'sample_length':100
    }
    
    dataset = gen_dataset(**config)
    for batch in dataset:
        print(batch.x_target.shape, batch.y_target.shape)



