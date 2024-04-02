import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import random
from numpy import random
from chunkllama_attn_replace import replace_with_chunkmistral
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_model():
    # pipeline parallelism
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", device_map="auto",
                                                trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval()
    return model


def generate_prompt_landmark(n_garbage, seed, percent):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_prefix = int(percent * n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 50000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(50000, 500000)

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


def passkey_retrieval_test(model, tokenizer, depth, use_cache=False, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, depth)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    len_token = input_ids.shape[-1]
    print("len tokens", len_token)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
    )
    print("[prediction]:  ", tokenizer.decode(generation_output[0][len_token:]))
    print("[ground truth]: ", answer)
    print("--------")

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    is_correct = (model_answer == answer_ids[0]).all().item()
    return is_correct, len_token


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default="mistral")
    parser.add_argument('--pretraining_length', type=int, default=32000)
    parser.add_argument('--scale', type=str, default="13b")
    parser.add_argument('--max_length', type=str, default="256k")
    parser.add_argument('--min_length', type=str, default="1k")
    parser.add_argument('--gap', type=str, default="8k")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--dca', action="store_true")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_config()
    if args.dca:
        replace_with_chunkmistral(args.pretraining_length)

    output_name = f"{args.model}.output.jsonl"
    print("results will be save to:", output_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = load_model()
    
    # hyper params
    k = 1000
    max_length = int(args.max_length.replace("k", '')) * k
    min_length = int(args.min_length.replace("k", '')) * k
    gap = int(args.gap.replace("k", '')) * k
    num_per = 16
    depth_percent = 1 / num_per
    
    # length_list = [k] + [i for i in range(4*k, max_length + 1, gap)]
    length_list = [i for i in range(min_length, max_length + 1, gap)]

    results = []
    for length in length_list:
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * length // 1024 * 1024)
 
        depths = [depth_percent * i for i in range(1, num_per + 1)]
        for depth in depths:
            passed_tests = 0
            all_accuries = {}
            for j in range(args.num_tests):
                torch.cuda.empty_cache()
                is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, depth, n_garbage=n_garbage, seed=j)
                passed_tests += is_correct
            accuracy = float(passed_tests) / args.num_tests
            res = {"context_length": f"{length // k}k", "depth_percent": depth * 100, "score": accuracy}
            results.append(res)
            print(res)
            with open(output_name, "a") as f:
                print(json.dumps(res), file=f)
