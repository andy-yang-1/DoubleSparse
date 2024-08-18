import os
import json


# llama-2-7b: 5.47
# model_path = "meta-llama/Llama-2-7b-hf"
# channel_path = "llama2-7b-channel-config.json"
# channel_path = "llama2-7b-qk-channel-config.json"

# llama-2-7b-chat: 6.94
# model_path = "meta-llama/Llama-2-7b-chat-hf"
# channel_path = "llama2-7b-chat-channel-config.json"
# channel_path = "llama2-7b-chat-qk-channel-config.json"

# llama-7b: 5.68
# model_path = "huggyllama/llama-7b"
# channel_path = "llama-7b-channel-config.json"
# channel_path = "llama-7b-qk-channel-config.json"

# vicuna-7b-v1.5-16k: 7.15
# model_path = "lmsys/vicuna-7b-v1.5-16k"
# channel_path = "vicuna-7b-v1.5-16k-channel-config.json"
# channel_path = "vicuna-7b-v1.5-16k-qk-channel-config.json"

# mistral-7b: 5.25
# model_path = "mistralai/Mistral-7B-v0.1"
# channel_path = "mistral-7b-v0.1-channel-config.json"
# channel_path = "mistral-7b-v0.1-qk-channel-config.json"

# llama-2-70b-chat: 4.65
# model_path = "meta-llama/Llama-2-70b-chat-hf"
# channel_path = "llama2-70b-chat-channel-config.json"
# channel_path = "llama2-70b-chat-qk-channel-config.json"

# llama-2-70b: 3.32
# model_path = "meta-llama/Llama-2-70b-hf"
# channel_path = "llama2-70b-channel-config.json"
# channel_path = "llama2-70b-qk-channel-config.json"

# mixtral-8x7B-v0.1: 3.84
# model_path = "mistralai/Mixtral-8x7B-v0.1"
# channel_path = "mixtral-8x7b-channel-config.json"
# channel_path = "mixtral-8x7b-qk-channel-config.json"



model_paths = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "huggyllama/llama-7b",
    "lmsys/vicuna-7b-v1.5-16k",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mixtral-8x7B-v0.1"
]

channel_paths1 = [
    "llama2-7b-channel-config.json",
    "llama2-7b-chat-channel-config.json",
    "llama-7b-channel-config.json",
    "vicuna-7b-v1.5-16k-channel-config.json",
    "mistral-7b-v0.1-channel-config.json",
    "llama2-70b-chat-channel-config.json",
    "llama2-70b-channel-config.json",
    "mixtral-8x7b-channel-config.json"
]

channel_paths2 = [
    "llama2-7b-qk-channel-config.json",
    "llama2-7b-chat-qk-channel-config.json",
    "llama-7b-qk-channel-config.json",
    "vicuna-7b-v1.5-16k-qk-channel-config.json",
    "mistral-7b-v0.1-qk-channel-config.json",
    "llama2-70b-chat-qk-channel-config.json",
    "llama2-70b-qk-channel-config.json",
    "mixtral-8x7b-qk-channel-config.json"
]

for i in range(len(model_paths)):
    model_path = model_paths[i]
    channel_path1 = channel_paths1[i]
    channel_path2 = channel_paths2[i]

    channel_config1 = json.load(open(channel_path1))
    channel_config2 = json.load(open(channel_path2))

    # merge channel_config1 and channel_config2
    channel_config = channel_config1
    for key in channel_config2:
        channel_config[key+".qk_proj"] = channel_config2[key]

    # save the merged channel_config into model_path, if path doesn't exist create the dir and file
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path+".json", "w") as f:
        json.dump(channel_config, f)