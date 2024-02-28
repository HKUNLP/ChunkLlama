# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple

import argparse
from LEval_config import *
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkllama_attn_replace import replace_with_chunkllama


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        actual_len = k_to_number(args.max_length)
        max_length = actual_len - max_new_tokens
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        sys_prompt = get_sys_prompt(args, file_name)
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, tokenizer) > max_length - 200:
                if "codeU" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt])  # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:])
                cnt += 100

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst}. Please directly give answer without any additional output or explanation "
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                    message += "\nAnswer:"
                else:
                    context = "Document is as follows. {document} Instruction: {inst} " + f"\nAnswer this question with {len(out.split())} words."
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long document>")
                inputs = tokenizer(text_inputs, return_tensors="pt").to(model.device)
                prompt_length = inputs.input_ids.size()[-1]
   
                sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                output = tokenizer.decode(sample[0][prompt_length:])

                save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['evaluation'] = d['evaluation']
                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.",
                                               "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(model.device)

                    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 100:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print(inst)
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
        fw.close()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric',
                        choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', default='exam_eval')
    parser.add_argument('--max_length', default="16k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--scale', type=str, default="7b")

    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--mc_tasks', action='store_true',
                        help='set this if you want to test all multiple choice tasks')
    args = parser.parse_args()

    replace_with_chunkllama(4096)
    model_path = f"meta-llama/llama-2-{args.scale}-chat-hf"

    if model_path[-1] == "/":
        model_path = model_path[:-1]

    open_source_model = model_path.split("/")[-1] + args.max_length
    pred_save_path = f"Predictions/{args.metric}/{open_source_model}"
    print(f"Your prediction file will be saved to: {pred_save_path}  , press enter to confirm...")
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

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, pred_save_path)
    sys.exit(main())

