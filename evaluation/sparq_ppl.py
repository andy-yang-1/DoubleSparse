import argparse
import os
import re
import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
import numpy as np

from fastchat.model import get_conversation_template
# from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from sparq_llama import convert_sparq


if __name__ == "__main__":

    # llama-2-7b: 5.47
    # model_path = "meta-llama/Llama-2-7b-hf"
    # channel_path = "llama2-7b-channel-config.json"
    # channel_path = "llama2-7b-qk-channel-config.json"

    # llama-2-7b-chat: 6.94
    model_path = "meta-llama/Llama-2-7b-chat-hf"

    # llama-7b: 5.68
    # model_path = "/home/ec2-user/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"

    # opt-6.7b: 10.86
    # model_path = "/home/ec2-user/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"

    # model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)

    # sparq
    k = 32
    r = 8
    model = convert_sparq(model, config, k, r)

    prompt = "Hello, my name is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    max_new_tokens = 2048-input_ids.shape[-1]

    output = model.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)[0]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]
    print(output)
