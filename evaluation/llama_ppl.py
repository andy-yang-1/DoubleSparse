import json
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from datasets import load_dataset
from functools import partial
import gc

from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config, change_llama_heavy_const
from offload_llama import convert_kvcache_llama_offloading, convert_llama_offloading_channel_config, change_llama_offloading_heavy_const

def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    torch.cuda.empty_cache()
    gc.collect()

    max_seq_len = 2048

    testenc = testenc.input_ids.to(model.device)
    print(testenc.shape)
    nsamples = testenc.shape[1] // max_seq_len
    model = model.eval()

    nlls = []

    # 57 -> nan

    gc.collect()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * max_seq_len):((i + 1) * max_seq_len)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * max_seq_len):((i + 1) * max_seq_len)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_seq_len
        nlls.append(neg_log_likelihood)
        batch = None
        torch.cuda.empty_cache()
        gc.collect()

    return torch.exp(torch.stack(nlls).sum() / (nsamples * max_seq_len))


models_configs = [
    {"model_path": "meta-llama/Llama-2-7b-chat-hf", "channel_configs": ["llama2-7b-chat-channel-config.json", "llama2-7b-chat-qk-channel-config.json"]},
    {"model_path": "meta-llama/Llama-2-7b-hf", "channel_configs": ["llama2-7b-channel-config.json", "llama2-7b-qk-channel-config.json"]},
    # {"model_path": "huggyllama/llama-7b", "channel_configs": ["llama-7b-channel-config.json", "llama-7b-qk-channel-config.json"]},
    # {"model_path": "lmsys/vicuna-7b-v1.5-16k", "channel_configs": ["vicuna-7b-v1.5-16k-channel-config.json", "vicuna-7b-v1.5-16k-qk-channel-config.json"]},
]



heavy_consts = [2048 // factor for factor in [4, 8, 16, 32]]
group_factors = [4, 8, 16, 32]

# heavy_consts = [2048 // factor for factor in [8, 16]]
# group_factors = [8, 16]

results = {}

for model_config in models_configs:
    model_path = model_config["model_path"]
    qk_config = model_config["channel_configs"][1]

    model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    original_score = evaluate(model, tokenizer).item()

    results[model_path] = {"original": original_score}

    # qk outlier
    with open(qk_config, "r") as f:
        channel_config = json.load(f)

    model = convert_kvcache_llama_offloading(model, config, 128, 4, 4, [0,1,31])
    model = convert_llama_offloading_channel_config(model, channel_config, "qk")

    for heavy_const in heavy_consts:
        for group_factor in group_factors:
            qbit = 4
            if group_factor < qbit:
                qbit = 16 // group_factor
                group_factor = 1
            else:
                group_factor = group_factor // qbit
            model = change_llama_offloading_heavy_const(model, heavy_const, group_factor, qbit)
            score = evaluate(model, tokenizer).item()
            print(f"qbit: {qbit}, heavy_const: {heavy_const}, group_factor: {group_factor}, score: {score}")
            results[model_path][f"qbit-{qbit}-heavy-{heavy_const}-group-{group_factor}-qk"] = score

    model = None
    gc.collect()
    torch.cuda.empty_cache()

with open("ppl-offloading.json", "w") as f:
    json.dump(results, f)


# sparsity_factors = [2, 4, 8, 16, 32, 64]
# qbits = [8, 4]

# results = {}

# for model_config in models_configs:
#     model_path = model_config["model_path"]
#     normal_config = model_config["channel_configs"][0]
#     qk_config = model_config["channel_configs"][1]

#     model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
#     tokenizer = LlamaTokenizer.from_pretrained(model_path)
#     config = AutoConfig.from_pretrained(model_path)

#     original_score = evaluate(model, tokenizer).item()

#     results[model_path] = {"original": original_score}

#     channel_config = None
#     with open(normal_config, "r") as f:
#         channel_config = json.load(f)

#     model = convert_kvcache_llama_heavy_recent(model, config, 128, 4, 4)

#     # q outlier
#     model = convert_llama_channel_config(model, channel_config, "q")

#     for qbit in qbits:
#         for sparsity_factor in sparsity_factors:
#             heavy_const = 2048 // sparsity_factor
#             group_factor = sparsity_factor * qbit // 16
#             if group_factor == 0 or group_factor >= 128:
#                 continue
#             model = change_llama_heavy_const(model, heavy_const, group_factor, qbit)
#             score = evaluate(model, tokenizer).item()
#             print(f"qbit: {qbit}, sparsity_factor: {sparsity_factor}, score: {score}")
#             results[model_path][f"qbit-{qbit}-sparsity-{sparsity_factor}-q"] = score

#     # k outlier
#     model = convert_llama_channel_config(model, channel_config, "k")

#     for qbit in qbits:
#         for sparsity_factor in sparsity_factors:
#             heavy_const = 2048 // sparsity_factor
#             group_factor = sparsity_factor * qbit // 16
#             if group_factor == 0 or group_factor >= 128:
#                 continue
#             model = change_llama_heavy_const(model, heavy_const, group_factor, qbit)
#             score = evaluate(model, tokenizer).item()
#             print(f"qbit: {qbit}, sparsity_factor: {sparsity_factor}, score: {score}")
#             results[model_path][f"qbit-{qbit}-sparsity-{sparsity_factor}-k"] = score

#     # qk outlier
#     with open(qk_config, "r") as f:
#         channel_config = json.load(f)
#     model = convert_llama_channel_config(model, channel_config, "qk")

#     for qbit in qbits:
#         for sparsity_factor in sparsity_factors:
#             heavy_const = 2048 // sparsity_factor
#             group_factor = sparsity_factor * qbit // 16
#             if group_factor == 0 or group_factor >= 128:
#                 continue
#             model = change_llama_heavy_const(model, heavy_const, group_factor, qbit)
#             score = evaluate(model, tokenizer).item()
#             print(f"qbit: {qbit}, sparsity_factor: {sparsity_factor}, score: {score}")
#             results[model_path][f"qbit-{qbit}-sparsity-{sparsity_factor}-qk"] = score

#     model = None
#     gc.collect()
#     torch.cuda.empty_cache()


# with open("wiki-2-ppl.json", "w") as f:
#     json.dump(results, f)
