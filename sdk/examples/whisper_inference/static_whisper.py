import types

import torch
from whisper.model import AudioEncoder, TextDecoder


# to whisper.encoder's head modules (conv1(Conv1d) and conv2(Conv1d))
class AudioEncoderHead(torch.nn.Module):
    def __init__(self, org_encoder: AudioEncoder) -> None:
        super().__init__()

        self.conv1 = org_encoder.conv1
        self.conv2 = org_encoder.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        return x


def encoder_self_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)

    return self.out(out)


def encoder_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.attn_ln(x))
    x = x + self.mlp(self.mlp_ln(x))
    return x


# whisper.encoder's module except to the head
class AudioEncoderTransformer(torch.nn.Module):
    def __init__(self, org_encoder: AudioEncoder) -> None:
        super().__init__()

        self.register_buffer("positional_embedding", org_encoder.positional_embedding)
        self.blocks = org_encoder.blocks
        self.ln_post = org_encoder.ln_post

        for block in self.blocks:
            # replace ResidualAttentionBlock.forward()
            block.forward = types.MethodType(encoder_block_forward, block)
            # replace MultiHeadAttention.forward()
            block.attn.forward = types.MethodType(encoder_self_attn_forward, block.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert (
            x.shape[1:] == self.positional_embedding.shape
        ), f"incorrect audio shape: {x.shape=} vs. {self.positional_embedding.shape=}"

        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


def static_self_attn_forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    offset: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]

    # Update the persistent KV cache in place. The cache tensors are registered
    # buffers of TextDecoderTransformer, so MLSDK keeps their device allocation
    # across decoder invocations.
    offset_idx = offset.view(1, 1, 1, 1).expand(*q.shape)
    k_cache.scatter_(dim=2, index=offset_idx, src=k)
    v_cache.scatter_(dim=2, index=offset_idx, src=v)

    a = torch.nn.functional.scaled_dot_product_attention(
        q,
        k_cache,
        v_cache,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)

    return self.out(out)


def static_cross_attn_forward(
    self,
    x: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    q = self.query(x)
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    # view() and permute() have been done in CrossKVPrecompute.
    k, v = kv_cache

    a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)

    return self.out(out)


def static_block_forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    offset: torch.Tensor | None = None,
    cross_kv_cache: (
        tuple[torch.Tensor, torch.Tensor] | None
    ) = None,  # should be precomputed
) -> torch.Tensor:
    # apply self_attention
    attn_out = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache, offset=offset)
    x = x + attn_out

    # apply cross_attention
    if self.cross_attn:
        x = x + self.cross_attn(self.cross_attn_ln(x), kv_cache=cross_kv_cache)

    # apply mlp
    x = x + self.mlp(self.mlp_ln(x))

    return x


# to calculation of the embeddings of the input tokens
class TextDecoderEmbedding(torch.nn.Module):
    def __init__(self, org_decoder: TextDecoder) -> None:
        super().__init__()
        self.token_embedding = org_decoder.token_embedding
        self.positional_embedding = org_decoder.positional_embedding

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        pos_emb = self.positional_embedding[offset[0]].view(
            1, 1, -1
        )  # to broadcast explictly
        x = self.token_embedding(x) + pos_emb

        return x


# whisper.decoder's module except to embeddings
class TextDecoderTransformer(torch.nn.Module):
    def __init__(self, org_decoder: TextDecoder, batch_size: int) -> None:
        super().__init__()

        self.blocks = org_decoder.blocks
        self.ln = org_decoder.ln
        self.token_embedding = org_decoder.token_embedding

        # TextDecoder does not expose n_ctx; its positional embedding has one
        # row per text-context position.
        cache_length = org_decoder.positional_embedding.shape[0] // 2
        n_head = self.blocks[0].attn.n_head
        n_state_per_head = self.token_embedding.embedding_dim // n_head
        cache_shape = (batch_size, n_head, cache_length, n_state_per_head)
        for i in range(len(self.blocks)):
            self.register_buffer(f"self_k_cache_{i}", torch.zeros(cache_shape))
            self.register_buffer(f"self_v_cache_{i}", torch.zeros(cache_shape))

        for block in self.blocks:
            # replace forward() of ResidualAttentionBlock
            block.forward = types.MethodType(static_block_forward, block)

            # replace forward() of self_attention
            block.attn.forward = types.MethodType(static_self_attn_forward, block.attn)

            # replace forward() of cross_attention
            block.cross_attn.forward = types.MethodType(
                static_cross_attn_forward, block.cross_attn
            )

    def self_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                getattr(self, f"self_k_cache_{i}"),
                getattr(self, f"self_v_cache_{i}"),
            )
            for i in range(len(self.blocks))
        ]

    def reset_self_kv_cache(self, context=None) -> None:
        """Clear self-attention state before decoding the next audio sample."""
        for k_cache, v_cache in self.self_kv_cache():
            k_cache.zero_()
            v_cache.zero_()
            if context is not None:
                context.get_registered_value_proxy(k_cache).load_from(
                    k_cache, clone=False
                )
                context.get_registered_value_proxy(v_cache).load_from(
                    v_cache, clone=False
                )

    def forward(
        self,
        x: torch.Tensor,  # embedded_tokens
        offset: torch.Tensor,  # offset for kv_cache
        mask: torch.Tensor,  # mask for kv_cache
        cross_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],  # len(list) == n_layer
    ) -> torch.Tensor:

        x = x.to(cross_kv_cache[0][0].dtype)

        self_kv_cache = self.self_kv_cache()
        for i, block in enumerate(self.blocks):
            self_kv = self_kv_cache[i]
            cross_kv = cross_kv_cache[i]
            x = block(
                x,
                mask=mask,
                kv_cache=self_kv,
                offset=offset,
                cross_kv_cache=cross_kv,
            )

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class CrossKVPrecompute(torch.nn.Module):
    """Precompute decoder cross-attention K/V caches from audio features."""

    def __init__(self, text_decoder_transformer: TextDecoderTransformer) -> None:
        super().__init__()

        # These projection modules are shared with TextDecoderTransformer.
        # Register TextDecoderTransformer with the MLSDK Context before compiling
        # the encoder so that their parameters are registered only once.
        self.keys = torch.nn.ModuleList(
            [block.cross_attn.key for block in text_decoder_transformer.blocks]
        )
        self.values = torch.nn.ModuleList(
            [block.cross_attn.value for block in text_decoder_transformer.blocks]
        )
        self.n_heads = [
            block.cross_attn.n_head for block in text_decoder_transformer.blocks
        ]

    def forward(self, audio_features: torch.Tensor) -> dict[str, torch.Tensor]:
        cross_kv_cache = {}

        for i, (key, value, n_head) in enumerate(
            zip(self.keys, self.values, self.n_heads)
        ):
            k = key(audio_features)
            v = value(audio_features)

            k = k.view(k.shape[0], k.shape[1], n_head, -1).permute(0, 2, 1, 3)
            v = v.view(v.shape[0], v.shape[1], n_head, -1).permute(0, 2, 1, 3)

            cross_kv_cache[f"cross_k_cache_{i}"] = k
            cross_kv_cache[f"cross_v_cache_{i}"] = v

        return cross_kv_cache
