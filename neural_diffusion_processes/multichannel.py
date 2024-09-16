from dataclasses import dataclass
from typing import Tuple
import math

import jax
import jax.numpy as jnp
import haiku as hk
from check_shapes import check_shapes
from check_shapes import check_shape as cs
from einops import rearrange, reduce

from .model import (
    MultiHeadAttention,
    timestep_embedding,
)


@dataclass
class AttentiveSqueezeAndExcite(hk.Module):
    num_heads: int
    hidden_dim: int
    apply_residual: bool = False

    @check_shapes(
        "s: [batch_size, channel, seq_len, input_dim, hidden_dim]",
        "return: [batch_size, channel, seq_len, input_dim, hidden_dim]",
    )
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        # global avg pool acros [N, D]
        squeezed = reduce(s, "b c n d h -> b c h", reduction="mean")
        mask = cs(
            jnp.zeros((s.shape[0], s.shape[1], s.shape[1]))[:, None, ...],
            "[batch_size, 1, channel, channel]",
        )
        attended_squeeze = MultiHeadAttention(self.hidden_dim, self.num_heads)(
            squeezed, squeezed, squeezed, mask
        )

        excitation_layer = SqueezeAndExcite(squeeze_over_axes=(2), inter_dim=s.shape[1] * 2)
        excite = excitation_layer(attended_squeeze)  # [B, C]

        out = s * excite[..., None, None, None]  # broadcast over input tensor
        return s + out if self.apply_residual else out


@dataclass
class SqueezeAndExcite(hk.Module):
    squeeze_over_axes: Tuple[int]
    inter_dim: int

    @check_shapes("s: [batch_size, channel, ...]", "return: [batch_size, channel]")
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        channels = s.shape[1]
        squeezed = jnp.mean(s, axis=self.squeeze_over_axes)

        reduction = jax.nn.gelu(
            hk.Linear(output_size=self.inter_dim)(squeezed)  # [B, R]
        )
        excitation = jax.nn.sigmoid(
            hk.Linear(output_size=channels)(reduction)  # [B, C]
        )

        return excitation


class MultiChannelBDAB(hk.Module):
    n_channels: int  # number of channels
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

        y_att_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_r, y_r, y_r, mask_type)
        y_att_n = cs(y_att_n, "[batch_size_x_channel, input_dim, num_points, hidden_dim_x2]")
        y_att_n = cs(
            jnp.swapaxes(y_att_n, 1, 2),
            "[batch_size_x_channel, num_points, input_dim, hidden_dim_x2]",
        )

        y_c = rearrange(
            jax.nn.gelu(y_att_n + y_att_d),
            "(b c) n d h -> b c n d h",
            c=self.n_channels,
            h=self.hidden_dim * 2,
        )
        y_att_c = AttentiveSqueezeAndExcite(
            self.num_heads, self.hidden_dim * 2, apply_residual=False
        )(y_c)

        y = rearrange(y_att_c, "b c n d h -> (b c) n d h", c=self.n_channels, h=self.hidden_dim * 2)
        residual, skip = jnp.split(y, 2, axis=-1)
        return (s + residual) / math.sqrt(2.0), skip


@dataclass
class MultiChannelBDAM(hk.Module):
    n_layers: int  # number of bdam and multi channel blocks
    n_channels: int  # number of channels
    hidden_dim: int
    num_heads: int
    init_zero: bool = True

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
        "x: [batch_size_x_channel, seq_len, input_dim]",
        "y: [batch_size_x_channel, seq_len, 1]",
        "t: [batch_size]",
        "mask_type: [...]",
        "return: [batch_size_x_channel, seq_len, 1]",
    )
    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        t: jnp.ndarray,
        mask_type: jnp.ndarray = jnp.array([[]]),
    ) -> jnp.ndarray:
        """
        Predicts the noise in each channel of input by interleaving the layers of BiDimensionalAttentionModel with a new ChannelAttentionLayer
        """

        x = cs(self.process_inputs(x, y), "[batch_size, num_points, input_dim, 2]")

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            "[batch_size, num_points, input_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        scale = math.sqrt(self.n_layers * 1.0)
        for _ in range(self.n_layers):
            layer = MultiChannelBDAB(self.n_channels, self.hidden_dim, self.num_heads)
            x, skip_con = layer(x, t_embedding, mask_type)  # [(B C) N D H]

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
