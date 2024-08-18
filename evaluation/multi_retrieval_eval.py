from transformers import BertTokenizerFast
import json

import gc

import random
import re
import torch
import wonderwords
from fastchat.model import load_model, get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


from modify_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
# from h2o_llama import convert_h2o, reset_h2o
from advanced_h2o_llama import convert_h2o, reset_h2o


def generate_with_kv_cache(model, input_ids, past_key_values=None, max_new_tokens=100):

    
    model_kwargs = {"past_key_values": past_key_values, "use_cache": True}

    gen_token = 0

    while gen_token < max_new_tokens:

        model_inputs = model.prepare_inputs_for_generation(input_ids, model_kwargs=model_kwargs)
        outputs = model(**model_inputs, use_cache=True, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token[:, None]], dim=-1)
        gen_token += 1

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs
        )
    
    return outputs, past_key_values


def generate_line_index(num_line):

    w = wonderwords.RandomWord()
    adjs = w.random_words(num_line, include_categories=["adjective"])
    nouns = w.random_words(num_line, include_categories=["noun"])

    comb = []
    for i, (adj, noun) in enumerate(zip(adjs, nouns)):
        comb.append(f"{adj}-{noun}")

    random_numbers = [random.randint(0, 50000) for _ in range(num_line)]
    
    return comb, random_numbers


def generate_lines_testcases(num_line):


    prompt_header = "Below is a record of lines I want you to remember. " + \
                    "Each line begins with 'line <line index>' and contains " + \
                    "a '<REGISTER_CONTENT>' at the end of the line as a numerical value. " + \
                    "For each line index, memorize its corresponding <REGISTER_CONTENT>. At " + \
                    "the end of the record, I will ask you to retrieve the corresponding " + \
                    "<REGISTER_CONTENT> of a certain line index. Now the record start:\n\n"

    lines = []
    line_idxes, random_numbers = generate_line_index(num_line)
    lines = [f"line {k}: REGISTER_CONTENT is <{v}>\n" for k, v in zip(line_idxes, random_numbers)]
    query = random.randint(0, len(line_idxes)-1)
    random_idx = line_idxes[query]
    answer = random_numbers[query]
    full_prompt = prompt_header + "".join(lines) + f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in line {random_idx}? I need the number."

    return line_idxes, random_numbers, full_prompt, answer


def two_rounds_testcases(model, tokenizer, num_line):

    line_idxes, random_numbers, full_prompt, answer = generate_lines_testcases(num_line)

    # First round
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    output = model.generate(input.input_ids.to("cuda"), max_new_tokens=100, use_cache=True)[0]
    # output, kv_cache = generate_with_kv_cache(model, input.input_ids, max_new_tokens=50)

    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {answer}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)


    # Second round

    second_query = random.randint(0, len(line_idxes)-1)
    random_idx = line_idxes[second_query]
    second_answer = random_numbers[second_query]

    second_prompt = f"Now tell me what is the <REGISTER_CONTENT> in line {random_idx}? I need the number."

    conv.update_last_message(output)
    conv.append_message(conv.roles[0], second_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    output = model.generate(input.input_ids.to("cuda"), max_new_tokens=100, use_cache=True)[0]
    # output, kv_cache = generate_with_kv_cache(model, input.input_ids, past_key_values=kv_cache, max_new_tokens=50)

    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    second_number = re.findall("\d+", output)
    if second_number is not None and len(second_number) > 0:
        second_number = int(second_number[-1])
    else:
        print(f"Got unparsable result")
        second_number = -1

    second_summary = f"Label: {second_answer}, Predict: {output}, Parsed: {second_number}, prompt length: {prompt_length}".replace('\n', ' ')  
    print(second_summary)  


    return (answer == response_number, second_answer == second_number), prompt_length, (summary, second_summary)



def multi_longeval_test(model, tokenizer, num_lines, num_tests):
    first_corrects = 0
    second_corrects = 0
    prompt_lengths = []
    summaries = []
    for _ in range(num_tests):
        (first_correct, second_correct), prompt_length, (first_summary, second_summary) = two_rounds_testcases(model, tokenizer, num_lines)
        first_corrects += first_correct
        second_corrects += second_correct
        prompt_lengths.append(prompt_length)
        summaries.append((first_summary, second_summary))
        reset_h2o(model)
    return first_corrects, second_corrects, prompt_lengths, summaries



# llama-2-7b-chat: 6.94
model_path = "meta-llama/Llama-2-7b-chat-hf"
# channel_path = "llama2-7b-chat-channel-config.json"
channel_path = "llama2-7b-chat-qk-channel-config.json"

# llama-7b: 5.68
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16"
# channel_path = "llama-7b-channel-config.json"
# channel_path = "llama-7b-qk-channel-config.json"

# vicuna-7b-v1.5-16k: 7.15
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5-16k/snapshots/9a93d7d11fac7f3f9074510b80092b53bc1a5bec"
# channel_path = "vicuna-7b-v1.5-16k-channel-config.json"
# channel_path = "vicuna-7b-v1.5-16k-qk-channel-config.json"

# opt-6.7b: 10.86
# model_path = "/home/ec2-user/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"

# model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

channel_config = None
with open(channel_path, "r") as f:
    channel_config = json.load(f)


# double sparsity
# model = convert_kvcache_llama_heavy_recent(model, config, 80, 4, 4)
# model = convert_llama_channel_config(model, channel_config, "qk")

# h2o
config.heavy_ratio = 0.8
config.recent_ratio = 0.1
model = convert_h2o(model, config)

first_corrects, second_corrects, prompt_lengths, summaries = multi_longeval_test(model, tokenizer, 50, 20)
print(first_corrects, second_corrects)




# line_idxes, random_numbers, full_prompt, answer = generate_lines_testcases(10)

# print(full_prompt)
    

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# text = full_prompt
# word = line_idxes[3]


# print("len(text):", len(text))

# tokens = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
# offset_mapping = tokens["offset_mapping"].squeeze()

# # print("mapping:", offset_mapping)

# start_char_pos = text.find(word)
# end_char_pos = start_char_pos + len(word)

# print("start_char_pos:", start_char_pos)
# print("end_char_pos:", end_char_pos)

# token_ids = []
# token_idxes = []
# for i, (start_pos, end_pos) in enumerate(offset_mapping):
#     if (start_pos >= start_char_pos and start_pos <= end_char_pos) or (start_char_pos >= start_pos and start_char_pos <= end_pos):
#         token_ids.append(tokens.input_ids[0][i].item())
#         token_idxes.append(i)

# print("Token IDs:", token_ids)
# print("Word: ", tokenizer.convert_ids_to_tokens(token_ids))