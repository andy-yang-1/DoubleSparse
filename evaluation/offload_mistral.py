# algo 1: use the upper layer's approximate attention
# algo 2: use the upper layer's embedding
# oracle: use the upper layer's full attention

import gc
import math
from typing import List, Optional, Tuple, Union
import types

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralAttention, apply_rotary_pos_emb, repeat_kv, MistralDecoderLayer, MistralModel



def pseudo_quantize(tensor, q_bit):
    max_quant = 2 ** q_bit - 1

    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    scale = max_quant / range_val
    quantized = torch.round((tensor - min_val) * scale).clamp(0, max_quant)

    dequantized = quantized / scale + min_val

    return dequantized


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=self.config.sliding_window,
    )
    hidden_states = inputs_embeds

    # if self.gradient_checkpointing and self.training:
    #     if use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #         )
    #         use_cache = False
    # decoder layers

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    upper_layer_embedding = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if idx in self.unstable_list:
            # change a lot here, can not inherant the upper layer's embedding
            upper_layer_embedding = None 
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)
                return custom_forward
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                upper_layer_embedding=upper_layer_embedding,
            )
        # algo 1: use the upper layer's approximate attention
        upper_layer_embedding = hidden_states
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    hidden_states = self.norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
    upper_layer_embedding: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
        upper_layer_embedding=upper_layer_embedding,
    )
    hidden_states = residual + hidden_states
    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs




class MistralAttention_offloading(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        upper_layer_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.config.num_hidden_layers != 32:
            gc.collect()
            torch.cuda.empty_cache()


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        # use previous embedding to compute approximate attention
        if upper_layer_embedding is not None:
            # print("upper_layer_embedding device:", upper_layer_embedding.device)
            # print("weight device:", self.q_proj.weight.device)
            previous_query_states = self.q_proj(upper_layer_embedding.to(hidden_states.device))
        else:
            previous_query_states = query_states
        # previous_query_states = query_states

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        previous_query_states = previous_query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        k = key_states
        query_states, key_states = apply_rotary_pos_emb(query_states, k, cos, sin, position_ids)
        
        # apply rotary to previous query states
        previous_query_states, _ = apply_rotary_pos_emb(previous_query_states, k, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


        # group_factor = 8
        assert self.head_dim % self.group_factor == 0

        if self.sorted_channel is not None:
            sorted_query_states = previous_query_states.transpose(1,2)
            sorted_key_states = key_states.transpose(1,2)
            sorted_query_states = torch.gather(sorted_query_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, q_len, -1, -1)).transpose(1,2)
            sorted_key_states = torch.gather(sorted_key_states, -1, self.sorted_channel.unsqueeze(0).unsqueeze(0).expand(bsz, kv_seq_len, -1, -1)).transpose(1,2)

            # grouped by mean
            # grouped_query = sorted_query_states.reshape(bsz, self.num_heads, q_len, self.head_dim // group_factor, group_factor).sum(dim=-1) / group_factor
            # grouped_key = sorted_key_states.reshape(bsz, self.num_heads, kv_seq_len, self.head_dim // group_factor, group_factor).sum(dim=-1) / group_factor
            # grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // group_factor)

            # outlier channel only
            outlier_num = self.head_dim // self.group_factor
            grouped_query = sorted_query_states[:,:,:,:outlier_num]
            grouped_key = sorted_key_states[:,:,:,:outlier_num]


            # quantization
            if self.label_bits < 16:
                grouped_query = pseudo_quantize(grouped_query, self.label_bits)
                grouped_key = pseudo_quantize(grouped_key, self.label_bits)


            grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)

            # precision problem??
        else:
            grouped_query = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
            grouped_key = key_states.reshape(bsz, self.num_heads, kv_seq_len, self.head_dim // self.group_factor, self.group_factor).sum(dim=-1) / self.group_factor
            grouped_attn_weights = torch.matmul(grouped_query, grouped_key.transpose(2, 3)) / math.sqrt(self.head_dim // self.group_factor)

        # assert torch.allclose(attn_weights, grouped_attn_weights, atol=0.001), f"{torch.nonzero(torch.abs(attn_weights - grouped_attn_weights))}"



        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            grouped_attn_weights = grouped_attn_weights + attention_mask


        h2_mask = torch.zeros_like(attn_weights).bool()
        # heavy_const = 256
        # [bs, num_heads, q_len, kv_len] -> [bs, num_heads, q_len, heavy_const]
        # sorted_weights, indices = attn_weights.sort(dim=-1, descending=True)
        _, indices = grouped_attn_weights.sort(dim=-1, descending=True)
        discard_indices = indices[:, :, :, self.heavy_const:]
        h2_mask.scatter_(3, discard_indices, 1)
        attn_weights.masked_fill_(h2_mask, float('-inf'))

        # free gpu memory
        if self.config.num_hidden_layers != 32:
            h2_mask = None
            grouped_attn_weights = None
            indices = None
            discard_indices = None
            grouped_query = None
            grouped_key = None
            sorted_query_states = None
            sorted_key_states = None
            query_states = None
            key_states = None
            gc.collect()
            torch.cuda.empty_cache()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def convert_kvcache_mistral_offloading(model, config, heavy_const=256, group_factor=8, label_bits=4, unstable_list=[0,1,31]):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_mistral_offloading(module, config, heavy_const, group_factor)

        if isinstance(module, MistralAttention):
            device = next(module.parameters()).device
            new_module = MistralAttention_offloading(config).half().to(device)
            new_module.load_state_dict(module.state_dict())
            new_module.heavy_const = heavy_const
            new_module.group_factor = group_factor
            new_module.label_bits = label_bits
            model._modules[name] = new_module

        if isinstance(module, MistralDecoderLayer):
            module.forward = layer_forward.__get__(module)
        
        if isinstance(module, MistralModel):
            module.forward = model_forward.__get__(module)
            module.unstable_list = unstable_list

    return model


def convert_mistral_offloading_channel_config(model, channel_config, selected_channel="k"):

    selected_channel = "." + selected_channel + "_proj"

    for name, module in model.named_modules():

        if isinstance(module, MistralAttention_offloading):
            device = next(module.parameters()).device
            module.sorted_channel = torch.tensor(channel_config[name + selected_channel]).to(device)

    return model


def change_mistral_offloading_heavy_const(model, heavy_const=128, group_factor=4, label_bits=4):

    for name, module in model.named_modules():

        if isinstance(module, MistralAttention_offloading):
            
            module.heavy_const = heavy_const
            module.group_factor = group_factor
            module.label_bits = label_bits

    return model