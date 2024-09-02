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
class ChannelConvolutionLayer(hk.Module):
 
    @check_shapes(
        's: [batch_size, channel, seq_len, input_dim, hidden_dim]',
        'return: [batch_size, channel, seq_len, input_dim, hidden_dim]'
    )
    def __call__(self, s: jnp.ndarray, ignore_alpha: bool = False) -> jnp.ndarray:
        kernelshape, hidden_dim = (self.kernel_depth, 1, 1), s.shape[-1]
        s = rearrange(s, 'batch_size, channel, seq_len, input_dim, hidden_dim -> batch_size, hidden_dim, channel, seq_len, input_dim')
        s_i = hk.Conv3D(hidden_dim, kernelshape, padding='SAME')(s)
        s_i = rearrange(jax.nn.gelu(s_i), 'batch_size, hidden_dim, channel, seq_len, input_dim -> batch_size, channel, seq_len, input_dim, hidden_dim')

        alpha = hk.get_parameter('alpha', [], init=jnp.zeros)
        return alpha * s + (1-alpha) * s_i if not ignore_alpha else s
    

@dataclass
class ChannelAttentionLayer(hk.Module):
    num_heads: int
    n_channels: int
    hidden_dim: int
    

    @check_shapes(
        "s: [batch_size, channel, seq_len, input_dim, hidden_dim]",
        "return[0]: [batch_size, channel, seq_len, input_dim, hidden_dim]",
        "return[1]: [batch_size, channel, seq_len, input_dim, hidden_dim]"
    )
    def __call__(self, s: jnp.ndarray, ignore_alpha: bool = False) -> jnp.ndarray:
        s = rearrange(s, 'batch_size channel seq_len input_dim hidden_dim -> batch_size seq_len input_dim channel hidden_dim')
        positional_encoding = timestep_embedding(jnp.arange(self.n_channels), embedding_dim=self.hidden_dim, max_positions=1_000)[None,None,None,...] # positional encodings [1, 1, 1, C, H]
        
        mask = jnp.zeros((s.shape[0], 1, 1, 1, self.n_channels, self.n_channels))
        attn_layer = MultiHeadAttention(self.hidden_dim*2, self.num_heads)
        s_enc = s 
        s_i = attn_layer(s_enc, s_enc, s_enc, mask_type=mask) #[B, N, D, C, Hx2]
        
        # map s_i back down to hidden_dim
        # s_i = cs(
        #     jax.nn.gelu(hk.Linear(self.hidden_dim)(s_i)),
        #     "[batch_size, seq_len, input_dim, channel, hidden_dim]"
        # )
        # implement same residual pattern as in BiDimensionalAttentionBlock

        # rearrange
        s = rearrange(s, "batch_size seq_len input_dim channel hidden_dim -> batch_size channel seq_len input_dim hidden_dim")
        s_i = rearrange(s_i, "batch_size seq_len input_dim channel hidden_dim_x_2 -> batch_size channel seq_len input_dim hidden_dim_x_2")
        res, skip = jnp.split(s_i, 2, axis=-1)

        alpha = hk.get_parameter('alpha', [], init=jnp.zeros)
        res = jax.nn.gelu(res)
        skip = jax.nn.gelu(skip)
        
        return (s + alpha*res) / math.sqrt(2.0), skip


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
        s_i += rearrange(positional_encoding, '(batch_size channel) 1 hidden_dim -> batch_size channel 1 hidden_dim', channel=n_channels)
        
        s_i = cs(hk.Conv2D(n_channels, kernelshape, padding='SAME', data_format='NCHW')(s_i), '[batch_size, channel, seq_len, hidden_dim]')

        alpha_1 = hk.get_parameter('alpha_1', shape=[], init=jnp.zeros)
        conv_result = alpha_1 * s + (1-alpha_1) * jax.nn.gelu(s_i) if not self.ignore_alpha else s # skip conv block if alpha = 1

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
        noise = cs(noise_fn(x, y, t, mask_type=mask_type), '[batch_size_x_channel, seq_len, hidden_dim]') # no masking by default
        # now the interleaving starts
        skip = None
        for _ in range(self.n_blocks):
            # rearrange here
            noise = rearrange(noise, '(batch_size channel) seq_len hidden_dim -> batch_size channel seq_len hidden_dim', channel=self.n_channels)
            noise = ChannelEncodingBlock(n_layers=self.n_layers, num_heads=self.num_heads, kernel_length=self.kernel_length, ignore_alpha=self.ignore_alpha)(noise)
            noise = rearrange(noise, 'batch_size channel seq_len hidden_dim -> (batch_size channel) seq_len hidden_dim', channel=self.n_channels)
            noise = hk.LayerNorm(-1, False, False)(noise) if skip is None else hk.LayerNorm(-1, False, False,)(noise)  + skip
            
            skip = jnp.copy(noise)

        
        out = noise/(math.sqrt(self.n_blocks))
        out = cs(
            jax.nn.gelu(hk.Linear(self.hidden_dim)(out)),
            '[batch_size_x_channel, seq_len, hidden_dim]'
        )
        return out
    

@dataclass
class MultiChannelBDAM(hk.Module):
    n_layers: int #number of bdam and multi channel blocks
    n_channels: int #number of channels
    hidden_dim: int
    num_heads: int
    init_zero: bool = True
    ignore_alpha: bool = False

    @check_shapes(
        "x: [batch_size, seq_len, input_dim]",
        "y: [batch_size, seq_len, output_dim]",
        "return: [batch_size, seq_len, input_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x dimensions for dimension-agnostic processing.
        """
        num_x_dims = jnp.shape(x)[-1]
        x = jnp.expand_dims(x, axis=-1)
        y = jnp.repeat(jnp.expand_dims(y, axis=-1), num_x_dims, axis=2)
        return jnp.concatenate([x, y], axis=-1)
    
    @check_shapes(
    'x: [batch_size_x_channel, seq_len, input_dim]',
    'y: [batch_size_x_channel, seq_len, 1]',
    't: [batch_size]',
    'mask_type: [...]',
    'return: [batch_size_x_channel, seq_len, 1]'
)
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, mask_type: jnp.ndarray = jnp.array([[]])) -> jnp.ndarray:
        '''
        Predicts the noise in each channel of input by interleaving the layers of BiDimensionalAttentionModel with a new ChannelAttentionLayer
        '''

        x = cs(self.process_inputs(x, y), "[batch_size, num_points, input_dim, 2]")

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            "[batch_size, num_points, input_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.n_layers):
            layer = BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
            x, skip_connection_1 = layer(x, t_embedding, mask_type) #[(B C), N, D, H]            
            
            # pass skip through the channel encoding block
            x =  rearrange(x, "(b c) n d h -> b c n d h", c=self.n_channels)
            skip = rearrange(skip_connection_1, "(b c) n d h -> b c n d h", c=self.n_channels)
            x, skip = ChannelAttentionLayer(self.num_heads, self.n_channels, self.hidden_dim)(x, self.ignore_alpha)
            x = rearrange(x, 'b c n d h -> (b c) n d h', c=self.n_channels)
            skip_connection_2 = rearrange(skip, 'b c n d h -> (b c) n d h', c=self.n_channels)

            skip = (skip_connection_1 + skip_connection_2) if skip is None else (skip_connection_1 + skip_connection_2) + skip

        x = cs(x, "[batch_size, num_points, input_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, input_dim, hidden_dim]")

        skip = cs(reduce(skip, "b n d h -> b n h", "mean"), "[batch, num_points, hidden_dim]")

        eps = skip / math.sqrt(self.n_layers * 2 * 1.0)
        eps_hidden = jax.nn.gelu(hk.Linear(self.hidden_dim)(eps))
        if self.init_zero:
            eps = hk.Linear(1, w_init=jnp.zeros)(eps_hidden)
        else:
            eps = hk.Linear(1)(eps_hidden)
        return eps
