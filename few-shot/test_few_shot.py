import json


def k_to_number(k_string):
    num = float(k_string.rstrip('k'))
    return int(num * 1000)


def num_tokens_from_string(string: str, tokenizer) -> int:
    encoding = tokenizer(string, return_tensors="pt")
    num_tokens = len(encoding['input_ids'][0])
    return num_tokens


def read_jsonl(train_fn):
    res = []
    with open(train_fn) as f:
        for i, line in enumerate(f):
            try:
                res.append(json.loads(line))
            except:
                continue
    print(f"loading from {train_fn}, there are {len(res)} samples")
    return res


import argparse
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkllama_attn_replace import replace_with_chunkllama


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    if args.task_path[-1] == "/":
        args.task_path = args.task_path[:-1]

    name = args.task_path.split("/")[-1]
    write_file = pred_save_path + name
    fw = open(write_file, "w")

    start_idx = 0
    for d in tqdm(data):
        document = d['input']
        cnt = 0
        while num_tokens_from_string(document, tokenizer) > max_length - 200:
            document = " ".join(document.split(" ")[cnt - max_length:])  # chunk the input len from left
            cnt += 100
        inst = d['query']
        out = d['output']
        query_examples = d["query_examples"]
        output_examples = d["output_examples"]

        save_d = {}
        save_d['query'] = inst
        save_d['gt'] = out
        text_inputs = document
        for i in range(len(query_examples)):
            text_inputs += f"\n\nQuestion: {query_examples[i]} \nAnswer:{output_examples[i]}"
        text_inputs += f"\n\nQuestion: {inst} \nAnswer:"
        inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.size()[-1]
        sample = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        output = tokenizer.decode(sample[0][prompt_length:], skip_special_tokens=True)
        v_list = output.split(f"Question:")
        for v_l in v_list:
            if len(v_l.replace("\n", '')) > 1:
                output = v_l
                break
        save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
        save_d["id"] = d["id"]
        save_d['pid'] = d['pid']
        if start_idx < 15:
            print("----------------- [output] vs [ground truth] -----------------")
            print(inputs.input_ids.shape)
            print(inst)
            print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
            start_idx += 1
        fw.write(json.dumps(save_d) + '\n')
    fw.close()
    samples = read_jsonl(write_file)
    pred_key = ""
    for key in samples[0]:
        if "pred" in key:
            pred_key = key
            break
    to_json = {}
    for sam in samples:
        to_json[sam["id"]] = sam[pred_key]
    with open(write_file.replace("jsonl", "json"), "w") as f:
        f.write(json.dumps(to_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str, required=True)
    parser.add_argument('--max_length', default="16k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--scale', default='13b', choices=['7b', '13b', '70b'])
    parser.add_argument('--max_new_tokens', default=64, type=int)

    args = parser.parse_args()

    replace_with_chunkllama(4096)

    if "qmsum" in args.task_path:
        args.max_new_tokens = 128

    model_path = f"meta-llama/llama-2-{args.scale}-hf"
    open_source_model = f"Chunkllama-{args.scale}" + args.max_length
    pred_save_path = f"Predictions/{open_source_model}/"

    max_length = k_to_number(args.max_length) - args.max_new_tokens
    data = read_jsonl(args.task_path)

    os.makedirs(pred_save_path, exist_ok=True)
    input(f"Your prediction file will be saved to: {pred_save_path}  , press enter to confirm...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if args.scale != "70b":
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
    else:
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
    model = model.eval()
    main()