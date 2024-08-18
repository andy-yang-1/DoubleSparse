import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
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
    input_dict = dict()
    output_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        # x_max [4096]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach().tolist()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]
        # feat [times, 4096]

    def stat_output_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            y = y[0]
        # x_max [4096]
        y_max = y.view(-1, y.shape[-1]).abs().mean(dim=0).cpu().detach().tolist()
        if name not in output_dict:
            output_dict[name] = [y_max]
        else:
            output_dict[name] += [y_max]
        # feat [times, 4096]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_output_max_hook, name=name)))

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return output_dict



model_path = "meta-llama/Llama-2-7b-hf"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"


model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
# model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer = LlamaTokenizer.from_pretrained(model_path)


input_feat = get_calib_feat(model, tokenizer)

# print(input_feat)

channels = [5, 1644, 2047, 3150, 4090]

fig, axes = plt.subplots(5, 2, figsize=(15, 20))  # 5x2 grid of plots

# name -> [times, 4096]
random_q_feat = np.array(list(input_feat.values())[0])
random_k_feat = np.array(list(input_feat.values())[1])
print(random_q_feat.shape)

for i, channel in enumerate(channels):
    ax = axes[i, 0]
    ax.hist(random_q_feat[:, channel], bins=50, alpha=0.75)
    ax.set_title(f"Q Channel {channel} Distribution")
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    ax = axes[i, 1]
    ax.hist(random_k_feat[:, channel], bins=50, alpha=0.75)
    ax.set_title(f"K Channel {channel} Distribution")
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')


# for i, (name, feat) in enumerate(input_feat.items()):
#     feat = np.array(feat)
#     ax = axes[i//2, i%2]
#     ax.plot(sum(feat))
#     ax.set_title(name)
#     if i == 9:
#         break

plt.savefig("llama_2_channel_distribution.png")