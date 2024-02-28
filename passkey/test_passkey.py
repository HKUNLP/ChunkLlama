import json
from typing import Optional, Tuple

from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import argparse
import random
from numpy import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkllama_attn_replace import replace_with_chunkllama


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 500000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, use_cache=False, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    len_token = input_ids.shape[-1]
    print("len tokens", len_token // 1000, "k")
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
    torch.cuda.empty_cache()
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
    )
    torch.cuda.empty_cache()
    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    is_correct = (model_answer == answer_ids[0]).all().item()
    print("answer", tokenizer.decode(generation_output[0][len_token:]))
    print(answer)
    if is_correct:
        print("success")
    else:
        print("fail")
    print("--------")
    return is_correct, len_token


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    if "70b" != args.scale:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights():
            model = LlamaForCausalLM.from_pretrained(model_path,
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(model, checkpoint=model_path,
                                             device_map='auto',
                                             offload_folder="./offload",
                                             no_split_module_classes=["LlamaDecoderLayer"],
                                             offload_state_dict=True, dtype=torch.bfloat16)
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(args.gpu)

    all_accuries = {}
    # This is a rough ratio to control the number of texts and tokens
    n_garbage = int(3.75 * args.seq_len // 1024 * 1024)
    passed_tests = 0
    total_tokens = 0
    for j in range(args.num_tests):
        torch.cuda.empty_cache()
        is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, n_garbage=n_garbage, seed=j)
        passed_tests += is_correct
        total_tokens += len_tokens
    avg_tokens = total_tokens // args.num_tests
    accuracy = float(passed_tests) / args.num_tests
    print("accuracy on the token length %d is %f" % (avg_tokens, accuracy))
    all_accuries["acc"] = accuracy
    all_accuries["local_window"] = args.local_window
    all_accuries["model"] = args.scale
    all_accuries["avg_tokens"] = avg_tokens
    with open("passkey.output.json", "a") as f:
        f.write(json.dumps(all_accuries)+"\n")
    print("acc over tokens", all_accuries)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--scale', type=str, default="13b")
    parser.add_argument('--seq_len', type=int, default=32768)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_tests', type=int, default=50, help='number of repeat testing for each length')
    parser.add_argument("--local_window", type=int, default=256, help='set to 64 for Llama 7B will lead to better results')
    parser.add_argument('--pretraining_length', type=int, default=4096)
    args = parser.parse_args()
    model_path = f"meta-llama/llama-2-{args.scale}-hf"
    replace_with_chunkllama(args.pretraining_length)
    main(args)
