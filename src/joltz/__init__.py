"""

Simple translation of Boltz-1 to JAX.

We use single-dispatch to convert PyTorch modules to Equinox modules recursively.

At a high-level we use the single-dispatch function `from_torch` to transform PyTorch modules to Equinox modules -- we use almost exactly the same names for the Equinox modules and their properties.

I use the convention that each equinox module has a static (or class) `from_torch` method that takes as its sole argument the corresponding PyTorch module and returns an instance of the equinox class.
Usually, this method calls `from_torch` on all of the pytorch module's children, and then constructs the equinox module using the results.
Because the implementation of this function is almost always the same I use `AbstractFromTorch` to define a default implementation of `from_torch`.
The final step is to register that function with the `from_torch` dispatcher. This is done with the `register_from_torch` class decorator, which takes a PyTorch module type and returns a decorator that registers the `from_torch` method of the equinox module.

`backend.py` contains the basic machinery for this and some translations of vanilla PyTorch modules.
This file contains translations of Boltz-1 modules.


"""
# TODO: This is basically a line-by-line translation: could make it more "equinox-y"
#   (e.g. no explicit batches, use dataclasses/eqx.Modules instead of dicts, use jaxtyping properly, etc)
# TODO: Chunking
# TODO: Model cache (?)
# TODO: Dropout

from dataclasses import fields
from functools import partial

import boltz
import boltz.model.layers.outer_product_mean
import boltz.model.layers.pair_averaging
import boltz.model.layers.transition
import boltz.model.layers.triangular_attention.attention
import boltz.model.layers.triangular_attention.primitives
import boltz.model.layers.triangular_mult
import boltz.model.model
import boltz.model.modules.confidence
import boltz.model.modules.diffusion
import boltz.model.modules.trunk
import boltz.model.modules.utils
import einops
import equinox as eqx
import jax
import numpy as np
from boltz.data import const

from jax import numpy as jnp
from jax import tree, vmap
from jaxtyping import Array, Bool, Float

from dataclasses import asdict

from .backend import (
    AbstractFromTorch,
    LayerNorm,
    Linear,
    from_torch,
    register_from_torch,
    Sequential,
    SparseEmbedding,
)


@from_torch.register(boltz.model.modules.utils.SwiGLU)
def _handle(_):
    def _swiglu(x):
        x, gates = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gates) * x

    return _swiglu


@register_from_torch(boltz.model.layers.transition.Transition)
class Transition(AbstractFromTorch):
    norm: LayerNorm
    fc1: Linear
    fc2: Linear
    fc3: Linear
    silu: callable

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... P"]:
        v = self.norm(x)
        return self.fc3(jax.nn.silu(self.fc1(v)) * self.fc2(v))


@register_from_torch(boltz.model.layers.pair_averaging.PairWeightedAveraging)
class PairWeightedAveraging(AbstractFromTorch):
    norm_m: LayerNorm
    norm_z: LayerNorm
    proj_m: Linear
    proj_g: Linear
    proj_z: Linear
    proj_o: Linear
    num_heads: int
    c_h: int  # channel heads
    inf: float

    def __call__(
        self,
        m: Float[Array, "B S N D"],
        z: Float[Array, "B N N D"],
        mask: Bool[Array, "B N N"],
    ) -> Float[Array, "B S N D"]:
        m = self.norm_m(m)
        z = self.norm_z(z)

        # follows the pytorch implementation -- could be rewritten in a more jax'ian (jaxish?) style

        # Project input tensors
        v = self.proj_m(m)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)  # no batch dimension
        v = jnp.transpose(v, (0, 3, 1, 2, 4))  # move heads to front, (b h s i d)

        # Compute weights
        b = self.proj_z(z)
        b = jnp.transpose(b, (0, 3, 1, 2))  # (b h i j)
        b = b + (1 - mask[:, None]) * -self.inf
        w = jax.nn.softmax(b, axis=-1)
        # Compute gating
        g = self.proj_g(m)
        g = jax.nn.sigmoid(g)

        # Compute output
        o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
        o = jnp.transpose(o, (0, 2, 3, 1, 4))
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        return self.proj_o(g * o)


@register_from_torch(boltz.model.layers.triangular_mult.TriangleMultiplicationOutgoing)
class TriangleMultiplicationOutgoing(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "N N D"], mask: Bool[Array, "N N"]
    ) -> Float[Array, "N N D"]:
        x = self.norm_in(x)
        x_in = x

        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))

        x = x * mask[..., None]

        a, b = jnp.split(x, 2, axis=-1)

        x = jnp.einsum("bikd,bjkd->bijd", a, b)

        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


@register_from_torch(boltz.model.layers.triangular_mult.TriangleMultiplicationIncoming)
class TriangleMultiplicationIncoming(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "N N D"], mask: Bool[Array, "N N"]
    ) -> Float[Array, "N N D"]:
        x = self.norm_in(x)
        x_in = x

        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))

        x = x * mask[..., None]

        a, b = jnp.split(x, 2, axis=-1)

        x = jnp.einsum("bkid,bkjd->bijd", a, b)

        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


@register_from_torch(boltz.model.layers.triangular_attention.primitives.Attention)
class Attention(AbstractFromTorch):
    c_q: int  # input dimension of query
    c_k: int  # input dimension of key
    c_v: int  # input dimension of value
    c_hidden: int  # per-head hidden dimension
    no_heads: int  # number of heads
    gating: bool  # whether to use gating
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: callable

    # TODO: Add mask? Instead of infs....
    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        biases: None | list[Float[Array, "... H Q K"]],
    ) -> Float[Array, "... Q C_v"]:
        # apply linear
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        # and reshape rearrange to heads (..., H, Q/K/V, C_hidden)
        q = einops.rearrange(
            q, "... Q (H C_hidden) -> ... H Q C_hidden", H=self.no_heads
        )
        k = einops.rearrange(
            k, "... K (H C_hidden) -> ... H K C_hidden", H=self.no_heads
        )
        v = einops.rearrange(
            v, "... V (H C_hidden) -> ... H V C_hidden", H=self.no_heads
        )

        # scale q
        q = q / np.sqrt(self.c_hidden)

        # compute attention
        a = jnp.einsum("... h q d, ... h k d -> ... h q k", q, k)
        # add pairwise biases
        # todo: not this.
        for bias in biases:
            a += bias

        a = jax.nn.softmax(a, axis=-1)

        a = jnp.einsum("... h q k, ... h k d -> ... h q d", a, v)
        # equivalent of o = o.transpose(-2, -3)
        o = einops.rearrange(a, "... H Q C_hidden -> ... Q H C_hidden")
        if self.linear_g is not None:
            g = jax.nn.sigmoid(self.linear_g(q_x))
            g = einops.rearrange(
                g, "... (H C_hidden) -> ... H C_hidden", H=self.no_heads
            )
            o = o * g

        o = einops.rearrange(o, "... Q H C -> ... Q (H C)")

        return self.linear_o(o)


# Boltz-1 has a triangle layer specific layernorm


@from_torch.register(boltz.model.layers.triangular_attention.primitives.LayerNorm)
def _trilayer_norm(m: boltz.model.layers.triangular_attention.primitives.LayerNorm):
    assert len(m.c_in) == 1
    return LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps)


@register_from_torch(
    boltz.model.layers.triangular_attention.attention.TriangleAttention
)
class TriangleAttention(AbstractFromTorch):
    c_in: int
    c_hidden: int
    no_heads: int
    starting: bool
    inf: float
    layer_norm: LayerNorm
    linear: Linear
    mha: Attention

    def __call__(
        self, x: Float[Array, "... I J C_in"], mask: Bool[Array, "... I J"]
    ) -> Float[Array, "... I J C_in"]:
        if not self.starting:
            x = einops.rearrange(x, "... I J C_in -> ... J I C_in")
            mask = einops.rearrange(mask, "... I J -> ... J I")

        x = self.layer_norm(x)

        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = einops.rearrange(self.linear(x), "... J I H -> ... 1 H J I")
        biases = [mask_bias, triangle_bias]
        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = einops.rearrange(x, "... J I C_in -> ... I J C_in")

        return x


@register_from_torch(boltz.model.layers.outer_product_mean.OuterProductMean)
class OuterProductMean(AbstractFromTorch):
    c_hidden: int
    norm: LayerNorm
    proj_a: Linear
    proj_b: Linear
    proj_o: Linear

    def __call__(
        self, m: Float[Array, "B S N D"], mask: Bool[Array, "B S N"]
    ) -> Float[Array, "B N N c_out"]:
        mask = mask[..., None]
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        # outer product mean
        mask = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask.sum(1).clip(min=1)
        z = jnp.einsum("bsic,bsjd->bijcd", a, b)
        z = einops.rearrange(z, "b i j c d -> b i j (c d)")
        z = z / num_mask
        return self.proj_o(z)


@register_from_torch(boltz.model.modules.trunk.MSALayer)
class MSALayer(AbstractFromTorch):
    msa_transition: Transition
    pair_weighted_averaging: PairWeightedAveraging
    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    z_transition: Transition
    outer_product_mean: OuterProductMean

    def __call__(self, z, m, token_mask, msa_mask):
        m = m + self.pair_weighted_averaging(m, z, token_mask)
        m = m + self.msa_transition(m)
        z = z + self.outer_product_mean(m, msa_mask)
        z = z + self.tri_mul_out(z, token_mask)
        z = z + self.tri_mul_in(z, token_mask)
        z = z + self.tri_att_start(z, token_mask)
        z = z + self.tri_att_end(z, token_mask)
        z = z + self.z_transition(z)

        return z, m


@register_from_torch(boltz.model.modules.trunk.MSAModule)
class MSAModule(eqx.Module):
    s_proj: Linear
    msa_proj: Linear
    stacked_params: MSALayer
    static: MSALayer

    def __call__(
        self,
        z: Float[Array, "B N N P"],
        emb: Float[Array, "B N D"],
        feats: dict[str, any],
    ) -> Float[Array, "B N N P"]:
        msa = feats["msa"]
        has_deletion = feats["has_deletion"][..., None]
        deletion_value = feats["deletion_value"][..., None]
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"]
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        m = jnp.concatenate([msa, has_deletion, deletion_value], axis=-1)
        m = self.msa_proj(m)
        m = m + jnp.expand_dims(self.s_proj(emb), 1)

        @jax.checkpoint
        def body_fn(embedding, params):
            return eqx.combine(self.static, params)(
                *embedding, token_mask, msa_mask
            ), None

        return jax.lax.scan(body_fn, (z, m), self.stacked_params)[0][0]

    @staticmethod
    def from_torch(module: boltz.model.modules.trunk.MSAModule):
        assert not module.use_paired_feature

        msa_layers = [MSALayer.from_torch(layer) for layer in module.layers]
        stacked_params = tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in msa_layers],
        )
        _, static = eqx.partition(msa_layers[0], eqx.is_inexact_array)

        return MSAModule(
            s_proj=from_torch(module.s_proj),
            msa_proj=from_torch(module.msa_proj),
            stacked_params=stacked_params,
            static=static,
        )


def _rearrange(pattern):
    return lambda x: einops.rearrange(x, pattern)


from_torch.register(einops.layers.torch.Rearrange, lambda x: _rearrange(x.pattern))


@register_from_torch(boltz.model.layers.attention.AttentionPairBias)
class AttentionPairBias(AbstractFromTorch):
    c_s: int  # input sequence dim
    num_heads: int
    head_dim: int
    inf: float
    initial_norm: bool
    norm_s: LayerNorm | None
    proj_q: Linear
    proj_k: Linear
    proj_v: Linear
    proj_g: Linear
    proj_z: Sequential
    proj_o: Linear

    def compute_pair_bias(self, z):
        return self.proj_z(z)

    # During diffusion sampling we can precompute pair bias from z
    def __call__(
        self,
        *,
        s: Float[Array, "B S D"],
        z: Float[Array, "B N N P"],
        mask: Bool[Array, "B N"],
        to_keys=None,
        bias=None,
    ):
        if bias is None:
            bias = self.compute_pair_bias(z)

        B = s.shape[0]
        assert s.ndim == 3

        if self.initial_norm:
            s = self.norm_s(s)
        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask[..., None])[..., 0]
        else:
            k_in = s

        q = self.proj_q(s).reshape(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).reshape(B, -1, self.num_heads, self.head_dim)

        g = jax.nn.sigmoid(self.proj_g(s))

        attn = jnp.einsum("bihd,bjhd->bhij", q, k)
        attn = attn / (self.head_dim**0.5) + bias
        attn = attn + (1 - mask[:, None, None]) * -self.inf
        attn = jax.nn.softmax(attn, axis=-1)
        o = jnp.einsum("bhij,bjhd->bihd", attn, v)
        o = o.reshape(B, -1, self.c_s)
        return self.proj_o(g * o)


# TODO: implement dropout
@register_from_torch(boltz.model.modules.trunk.PairformerLayer)
class PairformerLayer(AbstractFromTorch):
    token_z: int  # pairwise embedding dimension
    dropout: float
    num_heads: int
    no_update_s: bool
    no_update_z: bool
    attention: AttentionPairBias
    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    transition_s: Transition
    transition_z: Transition

    def __call__(
        self,
        s: Float[Array, "B N D"],
        z: Float[Array, "B N N P"],
        mask: Bool[Array, "B N"],
        pair_mask: Bool[Array, "B N N"],
    ):
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_att_start(z, pair_mask)
        z = z + self.tri_att_end(z, pair_mask)
        z = z + self.transition_z(z)
        assert not self.no_update_s
        s = s + self.attention(
            s=s, z=z, mask=mask, bias=self.attention.compute_pair_bias(z)
        )
        s = s + self.transition_s(s)

        return s, z


@register_from_torch(boltz.model.modules.trunk.PairformerModule)
class Pairformer(eqx.Module):
    stacked_parameters: PairformerLayer
    static: PairformerLayer

    @staticmethod
    def from_torch(m: boltz.model.modules.trunk.PairformerModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return Pairformer(
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )

    def __call__(self, s, z, mask, pair_mask):
        @jax.checkpoint
        def body_fn(embedding, params):
            return eqx.combine(self.static, params)(*embedding, mask, pair_mask), None

        return jax.lax.scan(body_fn, (s, z), self.stacked_parameters)[0]


@register_from_torch(boltz.model.modules.trunk.DistogramModule)
class Distogram(AbstractFromTorch):
    distogram: Linear

    def __call__(self, z: Float[Array, "B N N D"]) -> Float[Array, "B N N P"]:
        return self.distogram(z + z.transpose(0, 2, 1, 3))


@register_from_torch(boltz.model.modules.transformers.AdaLN)
class AdaLN(AbstractFromTorch):
    a_norm: LayerNorm
    s_norm: LayerNorm
    s_scale: Linear
    s_bias: Linear

    def __call__(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        return jax.nn.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


@register_from_torch(boltz.model.modules.transformers.ConditionedTransitionBlock)
class ConditionedTransitionBlock(AbstractFromTorch):
    adaln: AdaLN
    swish_gate: Sequential
    a_to_b: Linear
    b_to_a: Linear
    output_projection: Sequential

    def __call__(self, a, s):
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        return self.output_projection(s) * self.b_to_a(b)


@register_from_torch(boltz.model.modules.transformers.DiffusionTransformerLayer)
class DiffusionTransformerLayer(AbstractFromTorch):
    adaln: AdaLN
    pair_bias_attn: AttentionPairBias
    output_projection_linear: Linear
    output_projection: Sequential
    transition: ConditionedTransitionBlock

    def __call__(self, *, a, s, z, mask=None, to_keys=None, bias=None):
        b = self.adaln(a, s)

        assert a.ndim == 3

        # TODO: precompute pair bias
        b = self.pair_bias_attn(s=b, z=z, mask=mask, to_keys=to_keys, bias=bias)
        b = self.output_projection(s) * b
        a = a + b
        return a + self.transition(a, s)


@register_from_torch(boltz.model.modules.transformers.DiffusionTransformer)
class DiffusionTransformer(eqx.Module):
    stacked_parameters: DiffusionTransformerLayer
    static: DiffusionTransformerLayer

    def precompute_pair_biases(self, z):
        def precompute_layer_bias(params: DiffusionTransformerLayer):
            r = eqx.combine(params, self.static).pair_bias_attn.compute_pair_bias(z)
            return r

        return vmap(precompute_layer_bias)(self.stacked_parameters)

    def __call__(
        self, a, s, z, *, pair_biases=None, mask=None, to_keys=None, multiplicity=None
    ):
        if pair_biases is None:
            pair_biases = self.precompute_pair_biases(z)

        def body_fn(a, params_and_bias):
            # reconstitute layer
            params, bias = params_and_bias
            layer = eqx.combine(self.static, params)
            return layer(a=a, s=s, z=z, mask=mask, to_keys=to_keys, bias=bias), None

        return jax.lax.scan(body_fn, a, (self.stacked_parameters, pair_biases))[0]

    @staticmethod
    def from_torch(m: boltz.model.modules.transformers.DiffusionTransformer):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return DiffusionTransformer(
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )


@register_from_torch(boltz.model.modules.transformers.AtomTransformer)
class AtomTransformer(AbstractFromTorch):
    attn_window_queries: int
    attn_window_keys: int
    diffusion_transformer: DiffusionTransformer

    def precompute_pair_biases(self, q_shape, p):
        W = self.attn_window_queries
        H = self.attn_window_keys
        if W is not None:
            B, N, D = q_shape
            NW = N // W

            # p = p.reshape((p.shape[0] * NW, W, H, -1))
            # Hope this doesn't change...
            p = p.reshape((-1, 32, 128, 16))

        return self.diffusion_transformer.precompute_pair_biases(p)

    def __call__(self, q, c, p, to_keys=None, mask=None, cache=None, multiplicity=None):
        W = self.attn_window_queries
        H = self.attn_window_keys
        if W is not None:
            B, N, D = q.shape
            NW = N // W

            # reshape tokens
            q = q.reshape((B * NW, W, -1))
            c = c.reshape((B * NW, W, -1))
            if mask is not None:
                mask = mask.reshape(B * NW, W)
            p = p.reshape((p.shape[0] * NW, W, H, -1))

            to_keys_new = lambda x: to_keys(x.reshape(B, NW * W, -1)).reshape(
                B * NW, H, -1
            )
        else:
            to_keys_new = None

        assert cache is not None

        q = self.diffusion_transformer(
            a=q, s=c, z=p, mask=mask, to_keys=to_keys_new, pair_biases=cache
        )

        if W is not None:
            q = q.reshape((B, NW * W, D))
        return q


def get_indexing_matrix(K, W, H):
    # Just run this in torch and np the return array...
    return np.array(boltz.model.modules.encoders.get_indexing_matrix(K, W, H, "cpu"))


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.reshape(B, 2 * K, W // 2, D)
    r = jnp.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )
    return r


class AtomEncoderCache(eqx.Module):
    q: jax.Array
    c: jax.Array
    p: jax.Array
    to_keys: any
    transformer_cache: jax.Array


@register_from_torch(boltz.model.modules.encoders.AtomAttentionEncoder)
class AtomAttentionEncoder(AbstractFromTorch):
    embed_atom_features: Linear
    embed_atompair_ref_pos: Linear
    embed_atompair_ref_dist: Linear
    embed_atompair_mask: Linear
    atoms_per_window_queries: int
    atoms_per_window_keys: int

    structure_prediction: bool  # We should split this out into two different modules...

    c_to_p_trans_k: Sequential
    c_to_p_trans_q: Sequential
    p_mlp: Sequential
    atom_encoder: AtomTransformer
    atom_to_token_trans: Sequential

    s_to_c_trans: Sequential | None
    r_to_q_trans: Sequential | None
    z_to_p_trans: Sequential | None

    def compute_cache(self, feats, s_trunk, z=None):
        if z is None and self.structure_prediction:
            raise ValueError("z must be provided if structure_prediction is True")
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"]

        # assert model_cache is None

        atom_ref_pos = feats["ref_pos"]
        atom_uid = feats["ref_space_uid"]
        atom_feats = jnp.concatenate(
            [
                atom_ref_pos,
                feats["ref_charge"][..., None],
                atom_mask[..., None],
                feats["ref_element"],
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
            ],
            axis=-1,
        )

        c = self.embed_atom_features(atom_feats)
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = get_indexing_matrix(K, W, H)
        to_keys = partial(
            single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
        )

        atom_ref_pos_queries = atom_ref_pos.reshape(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).reshape(B, K, 1, H, 3)
        d = atom_ref_pos_keys - atom_ref_pos_queries
        d_norm = jnp.sum(d * d, axis=-1, keepdims=True)
        d_norm = 1 / (1 + d_norm)

        atom_mask_queries = atom_mask.reshape(B, K, W, 1)
        atom_mask_keys = to_keys(jnp.expand_dims(atom_mask, -1)).reshape(B, K, 1, H)
        atom_uid_queries = atom_uid.reshape(B, K, W, 1)
        atom_uid_keys = to_keys(jnp.expand_dims(atom_uid, -1)).reshape(B, K, 1, H)

        v = jnp.expand_dims(
            (
                jnp.logical_and(
                    jnp.logical_and(atom_mask_queries, atom_mask_keys),
                    (atom_uid_queries == atom_uid_keys),
                ).astype(jnp.float32)
            ),
            -1,
        )

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c

        if self.structure_prediction:
            atom_to_token = feats["atom_to_token"]

            s_to_c = self.s_to_c_trans(s_trunk)
            s_to_c = vmap(lambda M, v: M @ v)(atom_to_token, s_to_c)
            c = c + s_to_c

            atom_to_token_queries = atom_to_token.reshape(
                B, K, W, atom_to_token.shape[-1]
            )
            atom_to_token_keys = to_keys(atom_to_token)
            z_to_p = self.z_to_p_trans(z)
            z_to_p = jnp.einsum(
                "bijd,bwki,bwlj->bwkld",
                z_to_p,
                atom_to_token_queries,
                atom_to_token_keys,
            )
            p = p + z_to_p

        p = p + self.c_to_p_trans_q(c.reshape(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).reshape(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)

        transformer_cache = self.atom_encoder.precompute_pair_biases(q.shape, p)
        return AtomEncoderCache(q, c, p, to_keys, transformer_cache)

    def __call__(
        self,
        feats: dict[str, any],
        s_trunk=None,
        z=None,
        r=None,
        cache: AtomEncoderCache | None = None,
        multiplicity=1,
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"]

        q, c, p, to_keys = cache.q, cache.c, cache.p, cache.to_keys

        if self.structure_prediction:
            assert multiplicity == 1
            r_input = jnp.concatenate(
                [r, jnp.zeros((B * multiplicity, N, 7))],
                axis=-1,
            )
            r_to_q = self.r_to_q_trans(r_input)
            q = q + r_to_q

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            to_keys=to_keys,
            cache=cache.transformer_cache,
        )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = feats["atom_to_token"].astype(jnp.float32)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(axis=1, keepdims=True) + 1e-6
        )

        a = vmap(lambda M, v: M.T @ v)(atom_to_token_mean, q_to_a)

        return a, q, c, p, to_keys


@register_from_torch(boltz.model.modules.trunk.InputEmbedder)
class InputEmbedder(AbstractFromTorch):
    token_s: int
    no_atom_encoder: bool
    atom_attention_encoder: AtomAttentionEncoder

    def __call__(self, feats: dict[str, any]):
        # Load relevant features
        res_type = feats["res_type"]
        profile = feats["profile"]
        deletion_mean = feats["deletion_mean"][..., None]
        pocket_feature = feats["pocket_feature"]

        # Compute input embedding
        if self.no_atom_encoder:
            a = jnp.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
            )
        else:
            cache = self.atom_attention_encoder.compute_cache(feats, None, None)
            a, _, _, _, _ = self.atom_attention_encoder(feats, cache=cache)
        return jnp.concatenate(
            [a, res_type, profile, deletion_mean, pocket_feature], axis=-1
        )


@register_from_torch(boltz.model.modules.encoders.RelativePositionEncoder)
class RelativePositionEncoder(AbstractFromTorch):
    r_max: int
    s_max: int
    linear_layer: Linear

    def __call__(self, feats):
        b_same_chain = jnp.equal(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        b_same_residue = jnp.equal(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        b_same_entity = jnp.equal(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )
        d_residue = jnp.clip(
            feats["residue_index"][:, :, None]
            - feats["residue_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = jnp.where(
            b_same_chain, d_residue, jnp.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = jax.nn.one_hot(d_residue, 2 * self.r_max + 2)
        d_token = jnp.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = jnp.where(
            b_same_chain & b_same_residue,
            d_token,
            jnp.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = jax.nn.one_hot(d_token, 2 * self.r_max + 2)
        d_chain = jnp.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        d_chain = jnp.where(
            b_same_chain, jnp.zeros_like(d_chain) + 2 * self.s_max + 1, d_chain
        )
        a_rel_chain = jax.nn.one_hot(d_chain, 2 * self.s_max + 2)
        p = self.linear_layer(
            jnp.concatenate(
                [
                    a_rel_pos.astype(jnp.float32),
                    a_rel_token.astype(jnp.float32),
                    b_same_entity[..., None].astype(jnp.float32),
                    a_rel_chain.astype(jnp.float32),
                ],
                axis=-1,
            )
        )
        return p


@register_from_torch(boltz.model.modules.encoders.FourierEmbedding)
class FourierEmbedding(AbstractFromTorch):
    proj: Linear

    def __call__(self, times):
        times = einops.rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return jnp.cos(2 * jnp.pi * rand_proj)


@register_from_torch(boltz.model.modules.diffusion.OutTokenFeatUpdate)
class OutTokenFeatUpdate(AbstractFromTorch):
    sigma_data: float
    norm_next: LayerNorm
    fourier_embed: FourierEmbedding
    norm_fourier: LayerNorm
    transition_block: ConditionedTransitionBlock

    def __call__(self, times, acc_a, next_a):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = jnp.tile(
            self.norm_fourier(fourier_embed[:, None, ...]), (1, next_a.shape[1], 1)
        )
        cond_a = jnp.concatenate((acc_a, normed_fourier), axis=-1)
        acc_a = acc_a + self.transition_block(next_a, cond_a)
        return acc_a


@register_from_torch(boltz.model.modules.encoders.SingleConditioning)
class SingleConditioning(AbstractFromTorch):
    eps: float
    sigma_data: float
    norm_single: LayerNorm
    fourier_embed: FourierEmbedding
    norm_fourier: LayerNorm
    fourier_to_single: Linear
    transitions: list[Transition]
    single_embed: Linear

    def __call__(self, times, s_trunk, s_inputs):
        s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
        s = self.single_embed(self.norm_single(s))
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)
        s = einops.rearrange(fourier_to_single, "b d -> b 1 d") + s
        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


@register_from_torch(boltz.model.modules.encoders.PairwiseConditioning)
class PairwiseConditioning(AbstractFromTorch):
    dim_pairwise_init_proj: Sequential
    transitions: list[Transition]

    def __call__(self, z_trunk, token_rel_pos_feats):
        z = jnp.concatenate([z_trunk, token_rel_pos_feats], axis=-1)
        z = self.dim_pairwise_init_proj(z)
        for transition in self.transitions:
            z = transition(z) + z

        return z


class AtomDecoderCache(eqx.Module):
    transformer_cache: jax.Array


@register_from_torch(boltz.model.modules.encoders.AtomAttentionDecoder)
class AtomAttentionDecoder(AbstractFromTorch):
    a_to_q_trans: Linear
    atom_decoder: AtomTransformer
    atom_feat_to_atom_pos_update: Sequential

    def compute_cache(self, q_shape, p):
        return AtomDecoderCache(self.atom_decoder.precompute_pair_biases(q_shape, p))

    def __call__(
        self, *, a, q, c, p, feats, to_keys, cache: AtomDecoderCache, multiplicity=1
    ):
        assert multiplicity == 1
        atom_mask = feats["atom_pad_mask"]

        atom_to_token = feats["atom_to_token"]
        a_to_q = self.a_to_q_trans(a)
        a_to_q = vmap(lambda M, v: M @ v)(atom_to_token, a_to_q)
        q = q + a_to_q
        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            cache=cache.transformer_cache,
        )
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update


class DiffusionModelCache(eqx.Module):
    z: jax.Array
    token_transformer: jax.Array
    encoder_cache: AtomEncoderCache
    decoder_cache: AtomDecoderCache


@register_from_torch(boltz.model.modules.diffusion.DiffusionModule)
class DiffusionModule(AbstractFromTorch):
    atoms_per_window_queries: int
    atoms_per_window_keys: int
    sigma_data: int
    single_conditioner: SingleConditioning
    pairwise_conditioner: PairwiseConditioning
    atom_attention_encoder: AtomAttentionEncoder
    s_to_a_linear: Sequential
    token_transformer: DiffusionTransformer
    a_norm: LayerNorm
    atom_attention_decoder: AtomAttentionDecoder

    def compute_caches(
        self, feats, s_trunk, z_trunk, relative_position_encoding
    ) -> DiffusionModelCache:
        B, _, N = s_trunk.shape

        z = self.pairwise_conditioner(
            z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
        )
        encoder_cache = self.atom_attention_encoder.compute_cache(
            feats=feats, s_trunk=s_trunk, z=z
        )
        return DiffusionModelCache(
            token_transformer=self.token_transformer.precompute_pair_biases(z),
            encoder_cache=encoder_cache,
            z=z,
            decoder_cache=self.atom_attention_decoder.compute_cache(
                q_shape=(B, N, self.atoms_per_window_keys), p=encoder_cache.p
            ),
        )

    def __call__(
        self,
        *,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        cache: DiffusionModelCache,
        multiplicity=1,
    ):
        # s_inputs, s_trunk, and z_trunk are *constant* across calls to this module during diffusion sampling
        # anything expensive that just relies on these inputs should be precomputed
        assert multiplicity == 1
        s, normed_fourier = self.single_conditioner(
            times=times, s_trunk=s_trunk, s_inputs=s_inputs
        )

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=cache.z,
            r=r_noisy,
            multiplicity=multiplicity,
            cache=cache.encoder_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"]
        a = self.token_transformer(
            a,
            mask=mask,
            s=s,
            z=cache.z,
            multiplicity=multiplicity,
            pair_biases=cache.token_transformer,
        )
        a = self.a_norm(a)
        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            cache=cache.decoder_cache,
        )
        return {"r_update": r_update, "token_a": a}


def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return jnp.where(signs_differ, -a, a)


def random_quaternions(n: int, key):
    o = jax.random.normal(key=key, shape=(n, 4))
    s = (o * o).sum(1)
    o = o / _copysign(jnp.sqrt(s), o[:, 0])[:, None]
    return o


def quaternion_to_matrix(quaternions):
    r, i, j, k = jnp.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)

    o = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_rotations(n: int, key):
    """
    Generate random rotations as 3x3 rotation matrices.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """

    return quaternion_to_matrix(random_quaternions(n, key=key))


def randomly_rotate(coords, return_second_coords=False, second_coords=None, *, key):
    R = random_rotations(len(coords), key=key)

    if return_second_coords:
        return jnp.einsum("bmd,bds->bms", coords, R), (
            jnp.einsum("bmd,bds->bms", second_coords, R)
            if second_coords is not None
            else None
        )

    return jnp.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
    *,
    key,
):
    """Center and randomly augment the input coordinates."""

    if centering:
        atom_mean = jnp.sum(
            atom_coords * atom_mask[:, :, None], axis=1, keepdims=True
        ) / jnp.sum(atom_mask[:, :, None], axis=1, keepdims=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords, key=key
        )
        random_trans = (
            jax.random.normal(
                key=jax.random.fold_in(key, 1), shape=atom_coords[:, 0:1, :].shape
            )
            * s_trans
        )
        # random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights)[..., None]

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(axis=1, keepdims=True) / weights.sum(
        axis=1, keepdims=True
    )
    pred_centroid = (pred_coords * weights).sum(axis=1, keepdims=True) / weights.sum(
        axis=1, keepdims=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einops.einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant

    U, _, Vh = jnp.linalg.svd(cov_matrix, full_matrices=False)

    # Compute the rotation matrix
    rot_matrix = jnp.einsum("b i j, b j k -> b i k", U, Vh)

    # Ensure proper rotation matrix with determinant 1
    F = jnp.tile(jnp.eye(dim)[None], (batch_size, 1, 1))
    F = F.at[:, -1, -1].set(jnp.linalg.det(rot_matrix))
    rot_matrix = vmap(lambda U, F, Vh: U @ F @ Vh)(U, F, Vh)
    rot_matrix = jax.lax.stop_gradient(rot_matrix)

    # Apply the rotation and translation
    aligned_coords = (
        einops.einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )

    return aligned_coords


class StructureModuleOutputs(eqx.Module):
    sample_atom_coords: Float[Array, "B N 3"]
    diff_token_repr: Float[Array, "B N 768"]

@register_from_torch(boltz.model.modules.diffusion.AtomDiffusion)
class AtomDiffusion(AbstractFromTorch):
    score_model: DiffusionModule
    sigma_min: float
    sigma_max: FloatingPointError
    sigma_data: float
    rho: int
    P_mean: float
    P_std: float
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float
    coordinate_augmentation: bool
    alignment_reverse_diff: bool
    synchronize_sigmas: bool
    use_inference_model_cache: bool
    accumulate_token_repr: bool
    token_s: int
    out_token_feat_update: OutTokenFeatUpdate

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return jnp.log((sigma / self.sigma_data).clip(min=1e-20)) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        *,
        s_trunk,
        z_trunk,
        s_inputs,
        feats,
        relative_position_encoding,
        multiplicity: int = 1,
        model_cache: DiffusionModelCache,
    ):
        batch = noised_atom_coords.shape[0]
        sigma = jnp.full((batch,), sigma)

        padded_sigma = einops.rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
            multiplicity=multiplicity,
            cache=model_cache,
        )
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"]

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = (
            self.num_sampling_steps
            if num_sampling_steps is None
            else num_sampling_steps
        )
        inv_rho = 1 / self.rho

        steps = jnp.arange(num_sampling_steps)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = jnp.pad(
            sigmas, (0, 1), mode="constant", constant_values=0.0
        )  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps,
        multiplicity=1,
        *,
        key,
        s_trunk,
        z_trunk,
        s_inputs,
        feats,
        relative_position_encoding,
    ) -> StructureModuleOutputs:
        assert multiplicity == 1
        B, N, _ = s_trunk.shape

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = jnp.where(sigmas > self.gamma_min, self.gamma_0, 0.0)

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * jax.random.normal(key=key, shape=shape)
        atom_coords_denoised = jnp.zeros_like(atom_coords)

        token_repr = jnp.zeros((B, N, 768))
        token_a = jnp.zeros((B, N, 768))
        atom_coords, atom_coords_denoised = center_random_augmentation(
            atom_coords,
            atom_mask,
            augmentation=True,
            return_second_coords=True,
            second_coords=atom_coords_denoised,
            key=jax.random.fold_in(key, 0),
        )

        # State is
        # (atom_coords, atom_coords_denoised, token_repr, token_a)

        assert self.accumulate_token_repr
        assert self.alignment_reverse_diff

        # compute cache to avoid recomputing functions during repeat calls to score_model
        # TODO: This doesn't seem to save a whole lot of time...
        cache = self.score_model.compute_caches(
            feats=feats,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
        )

        def body_fn(state, sigmas_and_gammas):
            (atom_coords, atom_coords_denoised, token_repr, token_a) = state
            sigma_tm, sigma_t, gamma, key = sigmas_and_gammas
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
                key=jax.random.fold_in(key, 0),
            )
            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * jnp.sqrt(t_hat**2 - sigma_tm**2)
                * jax.random.normal(key=key, shape=shape)
            )
            atom_coords_noisy = atom_coords + eps
            atom_coords_denoised, token_a = self.preconditioned_network_forward(
                atom_coords_noisy,
                t_hat,
                multiplicity=multiplicity,
                model_cache=cache,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                s_inputs=s_inputs,
                feats=feats,
                relative_position_encoding=relative_position_encoding,
            )

            sigma = jnp.full(
                (atom_coords_denoised.shape[0],),
                t_hat,
            )
            token_repr = self.out_token_feat_update(
                times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
            )
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy,
                atom_coords_denoised,
                atom_mask,
                atom_mask,
            )
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next
            return (atom_coords, atom_coords_denoised, token_repr, token_a), None

        sigmas_and_gammas = (
            sigmas[:-1],
            sigmas[1:],
            gammas[1:],
            jax.random.split(key, num=num_sampling_steps),
        )

        state = (atom_coords, atom_coords_denoised, token_repr, token_a)
        state, _ = jax.lax.scan(body_fn, state, sigmas_and_gammas)

        return StructureModuleOutputs(sample_atom_coords=state[0], diff_token_repr=state[2])


def compute_aggregated_metric(logits, end=1.0):
    """Compute expected value of binned metric from logits"""
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = jnp.arange(start=0.5 * bin_width, stop=end, step=bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    plddt = einops.einsum(probs, bounds, "... b, b -> ...")

    return plddt


@register_from_torch(boltz.model.modules.confidence.ConfidenceHeads)
class ConfidenceHeads(AbstractFromTorch):
    max_num_atoms_per_token: int
    to_pde_logits: Linear
    to_plddt_logits: Linear
    to_resolved_logits: Linear
    to_pae_logits: Linear

    def __call__(self, s, z, x_pred, d, feats, pred_distogram_logits, multiplicity=1):
        assert multiplicity == 1
        plddt_logits = self.to_plddt_logits(s)
        assert len(z.shape) == 4
        pde_logits = self.to_pde_logits(z + z.transpose(0, 2, 1, 3))
        resolved_logits = self.to_resolved_logits(s)
        pae_logits = self.to_pae_logits(z)
        # Weights used to compute the interface pLDDT
        ligand_weight = 2
        interface_weight = 1

        # Retrieve relevant features
        token_type = feats["mol_type"]
        is_ligand_token = token_type == const.chain_type_ids["NONPOLYMER"]
        # Compute the aggregated pLDDT and iPLDDT
        plddt = compute_aggregated_metric(plddt_logits)
        token_pad_mask = feats["token_pad_mask"]
        complex_plddt = (plddt * token_pad_mask).sum(axis=-1) / token_pad_mask.sum(
            axis=-1
        )
        is_contact = d < 8
        is_different_chain = jnp.expand_dims(feats["asym_id"], -1) != jnp.expand_dims(
            feats["asym_id"], -2
        )

        token_interface_mask = jnp.max(
            is_contact
            * is_different_chain
            * jnp.expand_dims((1 - is_ligand_token), -1),
            axis=-1,
        )
        iplddt_weight = (
            is_ligand_token * ligand_weight + token_interface_mask * interface_weight
        )
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(axis=-1) / (
            jnp.sum(token_pad_mask * iplddt_weight, axis=-1) + 1e-5
        )
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = jax.nn.softmax(pred_distogram_logits, axis=-1)
        contacts = jnp.zeros((1, 1, 1, 64))
        contacts = contacts.at[:, :, :, :20].set(1.0)
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_pair_mask = (
            jnp.expand_dims(token_pad_mask, -1)
            * jnp.expand_dims(token_pad_mask, -2)
            * (1 - jnp.eye(token_pad_mask.shape[1])[None, ...])
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(axis=(1, 2)) / token_pair_mask.sum(
            axis=(1, 2)
        )
        asym_id = feats["asym_id"]
        token_interface_pair_mask = token_pair_mask * (
            jnp.expand_dims(asym_id, -1) != jnp.expand_dims(asym_id, -2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(axis=(1, 2)) / (
            token_interface_pair_mask.sum(axis=(1, 2)) + 1e-5
        )

        pae = compute_aggregated_metric(pae_logits, end=32)
        complex_pae = (pae * token_pair_mask).sum(axis=(1, 2)) / token_pair_mask.sum(
            axis=(1, 2)
        )

        complex_ipae = (pae * token_interface_pair_mask).sum(axis=(1, 2)) / (
            token_interface_pair_mask.sum(axis=(1, 2)) + 1e-5
        )

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
            complex_pae=complex_pae,
            complex_ipae=complex_ipae,
        )
        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        out_dict["i_pae"] = compute_aggregated_metric(pae_logits, end=32)
        # ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
        #     pae_logits, x_pred, feats, multiplicity
        # )
        # out_dict["ptm"] = ptm
        # out_dict["iptm"] = iptm
        # out_dict["ligand_iptm"] = ligand_iptm
        # out_dict["protein_iptm"] = protein_iptm
        # out_dict["pair_chains_iptm"] = pair_chains_iptm
        return out_dict


@register_from_torch(boltz.model.modules.confidence.ConfidenceModule)
class ConfidenceModule(eqx.Module):
    boundaries: Float[Array, "B"]
    dist_bin_pairwise_embed: SparseEmbedding
    confidence_heads: ConfidenceHeads
    # copies of most trunk modules
    s_init: Linear
    z_init_1: Linear
    z_init_2: Linear
    input_embedder: InputEmbedder
    rel_pos: RelativePositionEncoder
    token_bonds: Linear
    s_norm: LayerNorm
    z_norm: LayerNorm
    s_recycle: Linear
    z_recycle: Linear
    msa_module: MSAModule
    pairformer_module: Pairformer
    final_s_norm: LayerNorm
    final_z_norm: LayerNorm
    s_diffusion_norm: LayerNorm
    s_diffusion_to_s: Linear
    s_to_z: Linear
    s_to_z_transpose: Linear
    s_to_z_prod_in1: Linear
    s_to_z_prod_in2: Linear
    s_to_z_prod_out: Linear

    @staticmethod
    def from_torch(m: boltz.model.modules.confidence.ConfidenceModule):
        assert not m.no_update_s, "no_update_s not supported"
        assert m.use_s_diffusion, "use_s_diffusion must be True"
        assert m.add_s_to_z_prod, "add_s_to_z_prod must be True"
        assert m.imitate_trunk, "imitate_trunk must be True"

        return ConfidenceModule(
            **{k.name: from_torch(getattr(m, k.name)) for k in fields(ConfidenceModule)}
        )

    def __call__(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        feats,
        pred_distogram_logits,
        multiplicity,
        s_diffusion,
    ):
        assert multiplicity == 1
        s_inputs = self.input_embedder(feats)
        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"])
        # Apply recycling
        s = s_init + self.s_recycle(self.s_norm(s))
        z = z_init + self.z_recycle(self.z_norm(z))

        s_diffusion = self.s_diffusion_norm(s_diffusion)
        s = s + self.s_diffusion_to_s(s_diffusion)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        z = z + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )
        token_to_rep_atom = feats["token_to_rep_atom"]
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)

        assert len(token_to_rep_atom.shape) == 3
        x_pred_repr = vmap(lambda M, v: M @ v)(token_to_rep_atom, x_pred)
        d = jnp.linalg.norm(x_pred_repr[..., None, :] - x_pred_repr, axis=-1)
        distogram = (d[..., None] > self.boundaries).sum(axis=-1)
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram
        mask = feats["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        z = z + self.msa_module(z, s_inputs, feats)

        s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

        s, z = self.final_s_norm(s), self.final_z_norm(z)

        return self.confidence_heads(
            s=s,
            z=z,
            x_pred=x_pred,
            d=d,
            feats=feats,
            multiplicity=1,
            pred_distogram_logits=pred_distogram_logits,
        )



class TrunkState(eqx.Module):
    s: Float[Array, "B N 384"]
    z: Float[Array, "B N N D"]

class TrunkOutputs(eqx.Module):
    s: Float[Array, "B N 384"]
    z: Float[Array, "B N N D"]
    s_inputs: Float[Array, "B N P"]
    pdistogram: Float[Array, "B N N 64"]
    

@register_from_torch(boltz.model.model.Boltz1)
class Joltz1(eqx.Module):
    distogram_module: Distogram
    msa_module: MSAModule
    input_embedder: InputEmbedder
    s_init: Linear
    z_init_1: Linear
    z_init_2: Linear
    s_norm: LayerNorm
    z_norm: LayerNorm
    s_recycle: Linear
    z_recycle: Linear
    rel_pos: RelativePositionEncoder
    token_bonds: Linear
    pairformer_module: Pairformer
    structure_module: AtomDiffusion
    confidence_module: ConfidenceModule

    def embed_inputs(self, feats: dict) -> tuple[TrunkState, TrunkState, jax.Array]:
        with jax.default_matmul_precision("float32"):
            s_inputs = self.input_embedder(feats)
            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"])

            # Perform rounds of the pairwise stack
            s = jnp.zeros_like(s_init)
            z = jnp.zeros_like(z_init)

        return (TrunkState(s_init, z_init), TrunkState(s=s, z=z), s_inputs)

    def recycling_iteration(self, init: TrunkState, state: TrunkState, s_inputs, feats: dict):
        # One iteration of recyling
        with jax.default_matmul_precision("float32"):
            mask = feats["token_pad_mask"]
            pair_mask = mask[:, :, None] * mask[:, None, :]
            s = init.s + self.s_recycle(self.s_norm(state.s))
            z = init.z + self.z_recycle(self.z_norm(state.z))

            z = z + self.msa_module(z, s_inputs, feats)

            s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
            return TrunkState(s, z)
    
    def trunk(self, feats: dict, recycling_steps: int = 0) -> TrunkOutputs:
        with jax.default_matmul_precision("float32"):
            init, state, s_inputs = self.embed_inputs(feats)
            for _ in range(recycling_steps + 1):
                state = self.recycling_iteration(init, state, s_inputs, feats)
            return TrunkOutputs(state.s, state.z, s_inputs, self.distogram_module(state.z))

    def sample_structure(self, feats: dict, trunk_output: TrunkOutputs, num_sampling_steps: int = 25, key = None) -> StructureModuleOutputs:
        with jax.default_matmul_precision("float32"):
            return self.structure_module.sample(
                s_trunk=trunk_output.s,
                z_trunk=trunk_output.z,
                s_inputs=trunk_output.s_inputs,
                feats=feats,
                relative_position_encoding=self.rel_pos(feats),
                num_sampling_steps=num_sampling_steps,
                atom_mask=feats["atom_pad_mask"],
                multiplicity=1,
                key=key,
            )
    
    def predict_confidence(self, feats: dict, trunk_output: TrunkOutputs, structure_output: StructureModuleOutputs) -> dict:
        with jax.default_matmul_precision("float32"):
            return self.confidence_module(
                    s_inputs=trunk_output.s_inputs,
                    s=trunk_output.s,
                    z=trunk_output.z,
                    s_diffusion=structure_output.diff_token_repr,
                    x_pred=structure_output.sample_atom_coords,
                    feats=feats,
                    pred_distogram_logits=trunk_output.dict_out["pdistogram"],
                    multiplicity=1,
                )
                



    def __call__(
        self,
        feats: dict,
        recycling_steps: int = 0,
        num_sampling_steps: int = 25,
        sample_structure: bool = False,
        confidence_prediction: bool = False,
        key=None,
    ):
        # Legacy-style call method that returns unstructured dict
        trunk_embedding = self.trunk(feats, recycling_steps)
        dict_out = {"pdistogram": trunk_embedding.pdistogram}
        if sample_structure or confidence_prediction:
            structure_output = self.sample_structure(feats, trunk_embedding, num_sampling_steps, key)
            dict_out.update(asdict(structure_output))
        
        if confidence_prediction:
            dict_out.update(self.predict_confidence(feats, trunk_embedding, structure_output))
        
        return dict_out

    @staticmethod
    def from_torch(m: boltz.model.model.Boltz1):
        return Joltz1(
            **{k.name: from_torch(getattr(m, k.name)) for k in fields(Joltz1)}
        )
