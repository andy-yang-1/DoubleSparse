import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from datasets import load_dataset
from functools import partial
import tqdm
import matplotlib.pyplot as plt
import numpy as np

def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat(model, tokenizer):
    output_dict = dict()

    def stat_output_max_hook(m, x, y, name):
        if isinstance(y, tuple):
            y = y[0]
        # x_max [4096]
        #TODO support other models
        if y.shape[-1] != model.config.hidden_size:
            return
        y_max = y.view(-1, model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads).abs().mean(dim=0).cpu().detach()
        if name not in output_dict:
            output_dict[name] = y_max
        else:
            output_dict[name] += y_max
        # feat [times, 4096]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_output_max_hook, name=name)))

    print("Collecting activation scales...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return output_dict


@torch.no_grad()
def get_calib_qk_feat(model, tokenizer):
    output_dict = dict()

    def stat_qk_max_hook(m, args, kwargs, results, name):

        assert isinstance(kwargs, dict)
        # print(kwargs.keys())
        hidden_states = kwargs["hidden_states"]
        position_ids = kwargs["position_ids"]

        bsz, q_len, _ = hidden_states.size()

        q = m.q_proj(hidden_states).view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
        k = m.k_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        v = m.v_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
        kv_seq_len = k.shape[-2]
        cos, sin = m.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        k = repeat_kv(k, m.num_key_value_groups)
        v = repeat_kv(v, m.num_key_value_groups)

        out = q * k
        out = out.reshape(-1, m.num_heads, m.head_dim).abs().mean(dim=0).cpu().detach()

        if name not in output_dict:
            output_dict[name] = out
        else:
            output_dict[name] += out

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, LlamaAttention) or isinstance(m, MistralAttention) or isinstance(m, MixtralAttention):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_qk_max_hook, name=name), with_kwargs=True))

    print("Collecting activation scales...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return output_dict

# model_path = "meta-llama/Llama-2-7b-hf"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5-16k/snapshots/9a93d7d11fac7f3f9074510b80092b53bc1a5bec"
# model_path = "mistralai/Mistral-7B-v0.1"
# model_path = "meta-llama/Llama-2-70b-chat-hf"
# model_path = "meta-llama/Llama-2-70b-hf"
model_path = "mistralai/Mixtral-8x7B-v0.1"


kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
# model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer = LlamaTokenizer.from_pretrained(model_path)

# output_dict = get_calib_feat(model, tokenizer)
output_dict = get_calib_qk_feat(model, tokenizer)

channel_config = dict()

for k, v in output_dict.items():
    vals, inds = torch.sort(output_dict[k], dim=-1, descending=True)
    channel_config[k] = inds.tolist()

with open("mixtral-8x7b-qk-channel-config.json", "w") as f:
    json.dump(channel_config, f)