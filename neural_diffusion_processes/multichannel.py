from dataclasses import dataclass
from typing import Tuple
import math

import jax
import jax.numpy as jnp
import haiku as hk
from check_shapes import check_shapes
from check_shapes import check_shape as cs
from einops import rearrange, reduce

from .model import BiDimensionalAttentionModel, BiDimensionalAttentionBlock, MultiHeadAttention, timestep_embedding
from .process import EpsModel

@dataclass
class TriDimensionalAttentionBlock(hk.Module):
    hidden_dim: int
    num_heads: int

    @check_shapes(
        's: [batch_size, num_points, input_dim, output_dim, hidden_dim]',
        't: [batch_size, hidden_dim]',
        'mask_type: [...]',
        'return[0]: [batch_size, num_points, input_dim, output_dim, hidden_dim]',
        'return[1]: [batch_size, num_points, input_dim, output_dim, hidden_dim]'
    )
    def __call__(self, s: jnp.ndarray, t: jnp.ndarray, mask_type: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Tri-dimensional attention block - attends to output dimensions
        as well as inputs to generate a multivariate noise sample
        '''
        t = cs(
            hk.Linear(self.hidden_dim)(t)[:, None, None, :],
            "[batch_size, 1, 1, 1, hidden_dim]",
        )
        y = cs(s + t, "[batch_size, num_points, input_dim, output_dim, hidden_dim]")

        # num_points and input_dim part of batch dimension
        y_att_o = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y, y, y)
        y_att_o = cs(y_att_o, "[batch_size, num_points, input_dim, output_dim, hidden_dim_x2]")

        y_r = cs(jnp.swapaxes(y, 2, 3), '[batch_size, num_points, output_dim, input_dim, hidden_dim]')

        # now num_points and output_dim part of batch dimension
        y_att_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_r, y_r, y_r)
        y_att_d = cs(y_att_d, "[batch_size, num_points, output_dim, input_dim, hidden_dim_x2]")

        y_r = cs(jnp.swapaxes(y, 1, 3), "[batch_size, output_dim, input_dim, num_points, hidden_dim]")

        # now output_dim and input_dim part of batch dimension
        y_att_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_r, y_r, y_r, mask_type)
        y_att_n = cs(y_att_n, "[batch_size, output_dim, input_dim, num_points, hidden_dim_x2]")
        y_att_n = cs(
            jnp.swapaxes(y_att_n, 1, 3),
            "[batch_size, num_points, input_dim, output_dim, hidden_dim_x2]",
        )

        y = y_att_o + y_att_d + y_att_n

        residual, skip = jnp.split(y, 2, axis=-1)
        residual = jax.nn.gelu(residual)
        skip = jax.nn.gelu(skip)

        return (s + residual) / math.sqrt(2), skip
    

@dataclass
class TriDimensionalAttentionModel:
    n_layers: int
    """Number of bi-dimensional attention blocks."""
    hidden_dim: int
    num_heads: int
    init_zero: bool = True

    @check_shapes(
        "x: [batch_size, seq_len, input_dim]",
        "y: [batch_size, seq_len, output_dim]",
        "return: [batch_size, seq_len, input_dim, output_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x and y dimensions for dimension-agnostic processing.
        """
        D, O = x.shape[-1], y.shape[-1]

        new_x = jnp.repeat(jnp.expand_dims(x, -1), O, axis=3)
        new_y = jnp.repeat(jnp.expand_dims(y, -1), D, axis=3)
        new_y = jnp.swapaxes(new_y, -1, -2)

        return jnp.concatenate([new_x[..., None],new_y[..., None]], axis=-1)

    @check_shapes(
        "x: [batch_size, num_points, input_dim]",
        "y: [batch_size, num_points, output_dim]",
        "t: [batch_size]",
        "mask_type: [...]",
        "return: [batch_size, num_points, output_dim]",
    )
    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, mask_type: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the additive noise that was added to each dim of `y_0` to obtain `y_t`
        based on `x_t` and `y_t` and `t`
        """
        x = cs(self.process_inputs(x, y), "[batch_size, num_points, input_dim, output_dim, 2]")

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            "[batch_size, num_points, input_dim, output_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.n_layers):
            layer = TriDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
            x, skip_connection = layer(x, t_embedding, mask_type)
            skip = skip_connection if skip is None else skip_connection + skip

        x = cs(x, "[batch_size, num_points, input_dim, output_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, input_dim, output_dim, hidden_dim]")

        skip = cs(reduce(skip, "b n d o h -> b n o h", "mean"), "[batch, num_points, output_dim, hidden_dim]")

        eps = skip / math.sqrt(self.n_layers * 1.0)
        eps = jax.nn.gelu(hk.Linear(self.hidden_dim)(eps))
        if self.init_zero:
            eps = hk.Linear(1, w_init=jnp.zeros)(eps)
        else:
            eps = hk.Linear(1)(eps)
        return eps

@dataclass
class CrossChannelBiDimensionalAttentionModel(hk.Module):
    n_layers: int
    """Number of bi-dimensional attention blocks."""
    hidden_dim: int
    num_heads: int
    init_zero: bool = True

    """
    Extension of the bi-dimensional attention model with an added cross channel attention layer
    between clean and noisy inputs, so that multivariate prediction can take place without requiring
    multiple outputs for each input.

    The disadvantage is that, while agnostic to input length N and dimensionality D, it is
    not necessarily agnostic to output dimension O
    """

    @check_shapes(
        "x: [batch_size, seq_len, input_dim]",
        "y: [batch_size, seq_len, output_dim]",
        "return: [batch_size, seq_len, input_dim, output_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x and y dimensions for dimension-agnostic processing.
        """
        D, O = x.shape[-1], y.shape[-1]

        new_x = jnp.repeat(jnp.expand_dims(x, -1), O, axis=3)
        new_y = jnp.repeat(jnp.expand_dims(y, -1), D, axis=3)
        new_y = jnp.swapaxes(new_y, -1, -2)

        return jnp.concatenate([new_x[..., None],new_y[..., None]], axis=-1)
    
    
    def __call__(
        self,
        clean_x: jnp.ndarray,
        clean_y: jnp.ndarray,
        noisy_x: jnp.ndarray,
        noisy_y: jnp.ndarray,
        t: jnp.ndarray,
        mask_type: jnp.ndarray
    ) -> jnp.ndarray: pass

    
@dataclass
class ChannelConvolutionLayer(hk.Module):
    ''' number of filters '''
    kernel_depth: int
    ignore_alpha = False

    @check_shapes(
        's: [batch_size, channel, seq_len, input_dim, hidden_dim]',
        'return: [batch_size, channel, seq_len, input_dim, hidden_dim]'
    )
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        s_i = jnp.copy(s)
        kernelshape, padshape, hidden_dim = (self.kernel_depth, 1, 1), (1, 0, 0), s.shape[-1]
        s = rearrange(s, 'batch_size, channel, seq_len, input_dim, hidden_dim -> batch_size, hidden_dim, channel, seq_len, input_dim')
        s = hk.Conv3D(hidden_dim, kernelshape, padding=padshape)(s)
        s_i_1 = rearrange(s, 'batch_size, hidden_dim, channel, seq_len, input_dim -> batch_size, channel, seq_len, input_dim, hidden_dim')

        alpha_1 = hk.Bias(1)
        return alpha_1 * s_i + (1-alpha_1) * s_i_1 if not self.ignore_alpha else s_i
    

@dataclass
class ChannelAttentionLayer(hk.Module): pass


@dataclass
class ChannelEncodingBlock(hk.Module):
    n_layers: int
    num_heads: int
    kernel_length: int
    init_zero: bool = True
    ignore_alpha: bool = False # whether to not mix channel encoding with image

    @check_shapes(
       's: [batch_size, channel, seq_len, hidden_dim]',
       'return: [batch_size, channel, seq_len, hidden_dim]'     
    )
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:

        # 3d convolution first
        s_i = jnp.copy(s)
        kernelshape, padshape, n_channels = (1, self.kernel_length,), ((self.kernel_length-1)//2, 0), s.shape[1]

        positional_encoding = timestep_embedding(jnp.arange(n_channels), embedding_dim=s.shape[-1], max_positions=1_000) # generates positional encodings which need to be broadcasted across the channel dim
        positional_encoding = cs(
         jnp.tile(positional_encoding, (s.shape[0], 1))[...,None,:],
            '[batch_size_x_channel, 1, hidden_dim]'
        )
        s += rearrange(positional_encoding, '(batch_size channel) 1 hidden_dim -> batch_size channel 1 hidden_dim', channel=n_channels)
        
        s_i_1 = cs(hk.Conv2D(n_channels, kernelshape, padding='SAME', data_format='NCHW')(s), '[batch_size, channel, seq_len, hidden_dim]')

        alpha_1 = hk.get_parameter('alpha_1', shape=[], init=jnp.zeros)
        conv_result = alpha_1 * s_i + (1-alpha_1) * s_i_1 if not self.ignore_alpha else s_i
        conv_result = jax.nn.gelu(conv_result)

        # now multihead attn in channel direction
        s_i = jnp.copy(conv_result)
        s_i_1 = rearrange(
            conv_result,
            'batch_size channel seq_len hidden_dim -> batch_size seq_len channel hidden_dim'
        )

        s_i_1 = cs(
            MultiHeadAttention(d_model=s.shape[-1]*2, num_heads=self.num_heads)(s_i_1, s_i_1, s_i_1),
            '[batch_size, seq_len, channel, hidden_dim_x2]'
        )
        s_i_1 = hk.Linear(s_i.shape[-1])(s_i_1) # map back down to hidden_dim
        s_i_1 = rearrange(
            jax.nn.gelu(s_i_1),
            'batch_size seq_len channel hidden_dim -> batch_size channel seq_len hidden_dim'
        )                 
        alpha_2 = hk.get_parameter('alpha_2', shape=[], init=jnp.zeros)
        return alpha_2 * s_i + (1-alpha_2) * s_i_1 if not self.ignore_alpha else s_i


@dataclass
class MultiChannelEncodingModel(hk.Module):
    n_layers: int
    num_heads: int
    n_channels: int
    hidden_dim: int = 64
    n_blocks: int = 4
    init_zero: bool = True
    ignore_alpha: bool = False

    assert(hidden_dim >= 8)
    kernel_length = (hidden_dim // 8) + 1

    ''' Intended for use with a pre-trained BiDimensionalAttentionModel '''
    

    @check_shapes(
        'x: [batch_size_x_channel, seq_len, input_dim]',
        'y: [batch_size_x_channel, seq_len, 1]',
        't: [batch_size]',
        'mask_type: [...]',
        'return: [batch_size_x_channel, seq_len, hidden_dim]'
    )
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, mask_type: jnp.ndarray = jnp.array([[]])) -> jnp.ndarray:
        '''
        Predicts the noise in each channel of input by using a pretrained single-channel noise predictor and interleaving it with
        a new multi-channel encoding block.

        NOTE: model returns the latent representation of the noise, up to user to matmul with 'bi_dimensional_attention_model/linear_2'
        '''
        # this stays the same for each interleaved layer
        noise_fn = BiDimensionalAttentionModel(n_layers=self.n_layers, hidden_dim=self.hidden_dim, num_heads=self.num_heads, keep_hidden=True)
        # now the interleaving starts
        skip = None
        for _ in range(self.n_blocks):
            noise = cs(noise_fn(x, y, t, mask_type=mask_type), '[batch_size_x_channel, seq_len, hidden_dim]') # no masking by default
            # rearrange here
            noise = rearrange(noise, '(batch_size channel) seq_len hidden_dim -> batch_size channel seq_len hidden_dim', channel=self.n_channels)
            noise = ChannelEncodingBlock(n_layers=self.n_layers, num_heads=self.num_heads, kernel_length=self.kernel_length, ignore_alpha=self.ignore_alpha)(noise)
            noise = rearrange(noise, 'batch_size channel seq_len hidden_dim -> (batch_size channel) seq_len hidden_dim', channel=self.n_channels)
            noise = noise if skip is None else noise + skip
            
            skip = jnp.copy(noise)

        
        out = noise/(math.sqrt(self.n_blocks))
        out = cs(
            jax.nn.gelu(hk.Linear(self.hidden_dim)(out)),
            '[batch_size_x_channel, seq_len, hidden_dim]'
        )
        return out