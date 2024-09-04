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
        "s: [batch_size_x_channel, seq_len, input_dim, hidden_dim]",
        "return[0]: [batch_size_x_channel, seq_len, input_dim, hidden_dim]",
        "return[1]: [batch_size_x_channel, seq_len, input_dim, hidden_dim]"
    )
    def __call__(self, s: jnp.ndarray, ignore_alpha: bool = False) -> jnp.ndarray:
        s = rearrange(s, "(b c) n d h -> b n d c h", c=self.n_channels)

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
        s = rearrange(s, "b n d c h -> (b c) n d h", c=self.n_channels, h=self.hidden_dim)
        s_i = rearrange(s_i, "b n d c h -> (b c) n d h", c=self.n_channels, h=2*self.hidden_dim)
        res, skip = jnp.split(s_i, 2, axis=-1)

        # alpha = hk.get_parameter('alpha', [], init=jnp.zeros)
        res = jax.nn.gelu(res)
        skip = jax.nn.gelu(skip)
        
        return (s + res) / math.sqrt(2.0), skip
    
@dataclass
class MultiChannelBDAB(hk.Module):
    n_channels: int #number of channels
    hidden_dim: int
    num_heads: int
   
    @check_shapes(
        "s: [batch_size_x_channel, num_points, input_dim, hidden_dim]",
        "t: [batch_size_x_channel, hidden_dim]",
        "mask_type: [...]",
        "return[0]: [batch_size_x_channel, num_points, input_dim, hidden_dim]",
        "return[1]: [batch_size_x_channel, num_points, input_dim, hidden_dim]",
    )
    def __call__(
        self, s: jnp.ndarray, t: jnp.ndarray, mask_type: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bi-Dimensional Attention Block with an added MHSA layer that acts over the channel dimension 
        """
        t = cs(
            hk.Linear(self.hidden_dim)(t)[:, None, None, :],
            "[batch_size_x_channel, 1, 1, hidden_dim]",
        )
        y = cs(s + t, "[batch_size_x_channel, num_points, input_dim, hidden_dim]")

        # no mask needed as `num_points` is part of the batch dimension
        y_att_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y, y, y)
        y_att_d = cs(y_att_d, "[batch_size_x_channel, num_points, input_dim, hidden_dim_x2]")

        y_r = cs(jnp.swapaxes(y, 1, 2), "[batch_size_x_channel, input_dim, num_points, hidden_dim]")

        # if mask is not None:
        #     mask = jnp.expand_dims(mask, 1)

        y_att_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_r, y_r, y_r, mask_type)
        y_att_n = cs(y_att_n, "[batch_size_x_channel, input_dim, num_points, hidden_dim_x2]")
        y_att_n = cs(
            jnp.swapaxes(y_att_n, 1, 2),
            "[batch_size_x_channel, num_points, input_dim, hidden_dim_x2]",
        )
        
        # split dimensions and apply attention over channel dim
        y_c = rearrange(y, "(b c) n d h -> b n d c h", c=self.n_channels, h=self.hidden_dim)
        
        mask = jnp.zeros((y_c.shape[0], 1, 1, 1, self.n_channels, self.n_channels))
        y_att_c = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_c, y_c, y_c, mask)
        
        y_att_c = rearrange(
            y_att_c,
            "b n d c h -> (b c) n d h", 
            c=self.n_channels,
            h=2*self.hidden_dim
        )

        y = cs(y_att_n + y_att_d + y_att_c, "[batch_size_x_channel, num_points, input_dim, hidden_dim_x2]")

        residual, skip = jnp.split(y, 2, axis=-1)
        residual = jax.nn.gelu(residual)
        skip = jax.nn.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


@dataclass
class MultiChannelBDAM(hk.Module):
    n_layers: int #number of bdam and multi channel blocks
    n_channels: int #number of channels
    hidden_dim: int
    num_heads: int
    init_zero: bool = True
    use_channel_attention: bool = True #for ease of testing

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
        
        if self.use_channel_attention:
            skip = None
            scale = math.sqrt(self.n_layers * 2.0)
            for _ in range(self.n_layers):
                single_layer = BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
                x, skip1 = single_layer(x, t_embedding, mask_type) #[(B C), N, D, H]            
                
                channel_layer = ChannelAttentionLayer(self.num_heads, self.n_channels, self.hidden_dim)
                x, skip2 = channel_layer(x)
                
                skips = skip1 + skip2
                skip = skips if skip is None else  skip + skips

        else:
            skip = None
            scale = math.sqrt(self.n_layers * 1.0)
            for _ in range(self.n_layers):
                layer = MultiChannelBDAB(self.n_channels, self.hidden_dim, self.num_heads)
                x, skip_con = layer(x, t_embedding, mask_type) # [(B C) N D H]

                skip = skip_con if skip is None else skip + skip_con

        x = cs(x, "[batch_size, num_points, input_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, input_dim, hidden_dim]")

        skip = cs(reduce(skip, "b n d h -> b n h", "mean"), "[batch, num_points, hidden_dim]")

        eps = skip / scale
        eps_hidden = jax.nn.gelu(hk.Linear(self.hidden_dim)(eps))
        if self.init_zero:
            eps = hk.Linear(1, w_init=jnp.zeros)(eps_hidden)
        else:
            eps = hk.Linear(1)(eps_hidden)
        return eps

