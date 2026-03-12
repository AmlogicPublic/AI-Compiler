"""Static vision wrapper for Qwen3-VL export."""

import torch
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithDeepstackFeatures,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


class StaticQwen3VLVision(torch.nn.Module):
    """Fixed-shape vision module for torch.export/IREE."""

    def __init__(self, visual, image_grid_thw: torch.Tensor):
        super().__init__()
        self.config = visual.config
        self.spatial_merge_size = visual.spatial_merge_size
        self.patch_embed = visual.patch_embed
        self.blocks = visual.blocks
        self.deepstack_visual_indexes = visual.deepstack_visual_indexes
        self.deepstack_merger_list = visual.deepstack_merger_list
        self.merger = visual.merger

        with torch.no_grad():
            pos_embeds = visual.fast_pos_embed_interpolate(image_grid_thw)
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
            cu_seqlens = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        self.register_buffer("image_grid_thw", image_grid_thw)
        self.register_buffer("pos_embeds", pos_embeds)
        self.register_buffer("rotary_pos_emb", rotary_pos_emb)
        self.register_buffer("cu_seqlens", cu_seqlens)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def _attn_forward_static(self, attn_module, hidden_states, position_embeddings, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            attn_module.qkv(hidden_states)
            .reshape(seq_length, 3, attn_module.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_module.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, _ = attention_interface(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=attn_module.scaling,
            dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn_module.proj(attn_output)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        assert grid_thw.shape == self.image_grid_thw.shape
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.pos_embeds

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = self.rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = hidden_states + self._attn_forward_static(
                blk.attn,
                blk.norm1(hidden_states),
                position_embeddings,
                **kwargs,
            )
            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
