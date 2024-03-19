# -*- coding:utf-8 -*-
try:
    import fitz  # PyMuPDF
except ImportError:
    print("run: pip install PyMuPDF")

import os

import argparse
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from chunkllama_attn_replace import replace_with_chunkllama
from flash_attn_replace import replace_llama_attn_with_flash_attn

def load_model():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to(
        device)
    model = model.eval()
    return model

def parse_pdf2text(filename):
    try:
        doc = fitz.open(os.path.join(filename))
        text = ""
        for i, page in enumerate(doc):  # iterate the document pages
            text += f"<Page {i + 1}>: " + page.get_text()  # get plain text encoded as UTF-8
        print("read from: ", filename)
        sys_prompt =  "You are given a long paper. Please read the paper and answer the question.\n\n"
        return sys_prompt, text

    except:
        print("unable to parse", filename)
        return None


def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--scale', type=str, default="7b")
    parser.add_argument('--pdf', type=str, default="Popular_PDFs/chunkllama.pdf")
    parser.add_argument('--max_length', type=int, default=64000)
    args = parser.parse_args()
    return args


args = add_argument()

model_path = "togethercomputer/LLaMA-2-7B-32K"
tokenizer = LlamaTokenizer.from_pretrained(model_path, model_max_length=args.max_length, truncation_side="left",
                                           trust_remote_code=True)
# chunk attention
replace_with_chunkllama(pretraining_length=32768)
# original flash attention
# replace_llama_attn_with_flash_attn()

model = load_model()

sys_prompt, content = parse_pdf2text(args.pdf)

for i in range(100):
    question = input("User: ")
    message = sys_prompt + content  + f"Question:\n{question}"  + "\nAnswer:\n"
    prompt_length = tokenizer(message, return_tensors="pt").input_ids.size()[-1]
    if prompt_length > args.max_length:
        print("=" * 20)
        print(f"Your input length is {prompt_length}, and it will be truncated to {args.max_length}. You can set `--max_length` to a larger value ")
        print("=" * 20)
    inputs = tokenizer(message, truncation=True, return_tensors="pt").to(model.device)
    inp_length = inputs.input_ids.size()[-1]
    sample = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    output = tokenizer.decode(sample[0][inp_length:])
    print("Chatbot:",output)
    print(f"---------------End of round{i}------------------")
