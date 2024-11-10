from argparse import ArgumentParser
import json
import os

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.models.mllama.modeling_mllama import MllamaTextSelfAttention
from datasets import load_dataset
from functools import partial
import tqdm

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


def get_q_hook(m, x, y, name, output_dict, model):
    if isinstance(y, tuple):
        y = y[0]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert y.shape[-1] == model.config.hidden_size
    y_max = y.view(-1, model.config.num_attention_heads, head_dim).abs().mean(dim=0).cpu().detach()
    if name not in output_dict:
        output_dict[name] = y_max
    else:
        output_dict[name] += y_max
        
def get_k_hook(m, x, y, name, output_dict, model):
    if isinstance(y, tuple):
        y = y[0]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    assert y.shape[-1] == model.config.num_key_value_heads * head_dim
    y_max = y.view(-1, model.config.num_key_value_heads, head_dim).abs().mean(dim=0).cpu().detach()
    if name not in output_dict:
        output_dict[name] = y_max
    else:
        output_dict[name] += y_max
        
def get_qk_hook(m, args, kwargs, result, name, output_dict):
    assert isinstance(kwargs, dict)
    
    hidden_states = kwargs["hidden_states"]
    position_ids = kwargs["position_ids"]

    bsz, q_len, _ = hidden_states.size()

    q = m.q_proj(hidden_states).view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
    k = m.k_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
    v = m.v_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
    kv_seq_len = k.shape[-2]
    if isinstance(m, LlamaAttention):
        cos, sin = m.rotary_emb(v, position_ids)
        q, k = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
    elif isinstance(m, MixtralAttention):
        cos, sin = m.rotary_emb(v, kv_seq_len)
        q, k = transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    elif isinstance(m, MllamaTextSelfAttention):
        cos, sin = kwargs["position_embeddings"]
        q, k = transformers.models.mllama.modeling_mllama.apply_rotary_pos_emb(q, k, cos, sin)
    
    k = repeat_kv(k, m.num_key_value_groups)
    v = repeat_kv(v, m.num_key_value_groups)
    
    # Method 1: every token only attend to itself
    out = q * k
    out = out.reshape(-1, m.num_heads, m.head_dim).abs().mean(dim=0).cpu().detach()
    
    # Method 2: every token attend to all tokens and get the abs mean
    # out = torch.einsum('bhik,bhjk->bhijk', q, k) # [bsz, num_heads, kv_seq_len, kv_seq_len, head_dim]
    # out = out.abs().mean(dim=3).mean(dim=2).mean(dim=0).cpu().detach()
    
    # Method 3: every token attend to all tokens with mask and use softmax
    # out = torch.einsum('bhik,bhjk->bhijk', q, k) # [bsz, num_heads, kv_seq_len, kv_seq_len, head_dim]
    # boolean_mask = torch.tril(torch.ones(kv_seq_len, kv_seq_len, dtype=torch.bool, device=out.device))
    # mask = torch.zeros(kv_seq_len, kv_seq_len, dtype=torch.float16, device=out.device)
    # mask = mask.masked_fill(boolean_mask == False, float('-inf'))
    # out += mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    # out = out.reshape(bsz, m.num_heads, kv_seq_len, kv_seq_len * m.head_dim)
    # attn_score = torch.softmax(out, dim=-1) # Compute the contribution of each channel
    # attn_score = attn_score.reshape(bsz, m.num_heads, kv_seq_len, kv_seq_len, m.head_dim)
    # out = attn_score.mean(dim=3).mean(dim=2).mean(dim=0).cpu().detach()
    
    # Method 4: per channel contribution to attention scores
    # out = torch.einsum('bhik,bhjk->bhijk', q, k) # [bsz, num_heads, kv_seq_len, kv_seq_len, head_dim]
    # out = out.sum(dim=3).mean(dim=2).mean(dim=0).abs().cpu().detach()
    
    # Method 5: chanel ranking based on contribution to attention score variance
    # out = torch.einsum('bhik,bhjk->bhijk', q, k) # [bsz, num_heads, kv_seq_len, kv_seq_len, head_dim]
    # variance = out.var(dim=3).mean(dim=2).mean(dim=0)
    # out = variance.cpu().detach()
    
    
    if name not in output_dict:
        output_dict[name+".qk_proj"] = out
    else:
        output_dict[name+".qk_proj"] += out



@torch.no_grad()
def get_calib_feat(model: nn.Module, tokenizer):
    output_dict = dict()

    hooks = []
    for name, m in model.named_modules():
        # get_qkv hook
        if isinstance(m, nn.Linear) and "q_proj" in name:
            hooks.append(
                m.register_forward_hook(
                    partial(get_q_hook, name=name, output_dict=output_dict, model=model)))
        if isinstance(m, nn.Linear) and "k_proj" in name:
            hooks.append(
                m.register_forward_hook(
                    partial(get_k_hook, name=name, output_dict=output_dict, model=model)))
        # attention hook
        if isinstance(m, LlamaAttention) or isinstance(m, MistralAttention) or isinstance(m, MixtralAttention) or isinstance(m, MllamaTextSelfAttention):
            hooks.append(
                m.register_forward_hook(
                    partial(get_qk_hook, name=name, output_dict=output_dict), with_kwargs=True))

    print("Collecting activation scales...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    samples = get_calib_dataset(tokenizer, n_samples=256, block_size=512)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return output_dict


if __name__ == '__main__':
    parser = ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_path', type=str, required=True, help='Selected model')
    parser.add_argument('--output_dir', type=str, default="config/", help='Output directory')
    
    args = parser.parse_args()
    
    
    model_path = args.model_path
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    output_dict = get_calib_feat(model, tokenizer)

    channel_config = dict()

    for k, v in output_dict.items():
        vals, inds = torch.sort(output_dict[k], dim=-1, descending=True)
        channel_config[k] = inds.tolist()
        
    model_owner, model_name = model_path.split("/")
    
    output_dir = os.path.join(args.output_dir, model_owner)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, model_name + ".json")

    with open(output_path, "w") as f:
        json.dump(channel_config, f)