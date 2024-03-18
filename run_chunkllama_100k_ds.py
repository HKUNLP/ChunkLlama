# -*- coding:utf-8 -*-
try:
    import fitz  # PyMuPDF
except ImportError:
    print("run: pip install PyMuPDF")

import os

import argparse
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import deepspeed
import torch.distributed as dist
from newchunkllama_attn_replace import replace_with_chunkllama
from flash_attn_replace import replace_llama_attn_with_flash_attn

def load_model():
    model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16, device_map="cpu")
    config = {
        "dtype": model.dtype,
        "tensor_parallel": {
            "enabled": True, "tp_size": int(os.environ.get('WORLD_SIZE', '1'))
        },
        "max_tokens":250000,
        "quant": {"enabled": False},
        "injection_policy": {LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
    }
    model = deepspeed.init_inference(model=model, config=config)
    print("Model loaded successfully.")
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
    parser.add_argument('--scale', type=str, default="13b")
    parser.add_argument('--pdf', type=str, default = "Popular_PDFs/chunkllama.pdf")
    parser.add_argument('--max_length', type=int, default=32000)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


args = add_argument()


model_path = f"meta-llama/llama-2-{args.scale}-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_path, model_max_length=args.max_length, truncation_side="left", trust_remote_code=True)
# chunk attention
replace_with_chunkllama(pretraining_length=4096)
# original flash attention
# replace_llama_attn_with_flash_attn()
model = load_model()
sys_prompt, content = parse_pdf2text(args.pdf)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
for i in range(100):
    if args.local_rank ==0:
        question = input("User: ")
    else:
        question = None
    dist.barrier()
    message = B_INST + B_SYS + sys_prompt + E_SYS + content + f"Question:\n{question}" + E_INST + "Answer:\n"
    prompt_length = tokenizer(message, return_tensors="pt").input_ids.size()[-1]
    if prompt_length > args.max_length and args.local_rank == 0:
        print("="*20)
        print(f"Your input length is {prompt_length}, and it will be truncated to {args.max_length}. You can set `--max_length` to a larger value ")
        print("="*20)
    
    inputs = tokenizer(message, truncation=True, return_tensors="pt").to("cuda")
    inp_length = inputs.input_ids.size()[-1]
    sample = model.generate(**inputs, do_sample=False, max_new_tokens=32)
    output = tokenizer.decode(sample[0][inp_length:])
    if args.local_rank == 0:
        print("----------------model response-----------------")
        print("Chatbot:", output)
        print(f"---------------End of round{i}------------------")
