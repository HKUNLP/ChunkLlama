# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple

import argparse
from tqdm import tqdm
import numpy as np

import torch
from transformers import LlamaForCausalLM

import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkllama_attn_replace import replace_with_chunkllama

def get_as_batch(data, seq_length, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    # example [0, 256, 512]
    all_ix.pop()
    for idx in all_ix:         
        x = torch.stack([torch.from_numpy((data[idx:idx + seq_length]).astype(np.int64))])
        if device != 'cpu':
            x = x.pin_memory().to(device, non_blocking=True)
        yield x


def iceildiv(x, y):
    return (x + y - 1) // y


def evaluate_ppl_all(seq_length=8192, sliding_window=256, use_cache=False, args=None, model=None, data=None):
    model.eval()
    total_loss = []
    print(f"Test PPL on seq length {seq_length}")
    torch.set_printoptions(sci_mode=False)
    for idx, x in tqdm(
            enumerate(
                get_as_batch(
                    data['val'],
                    seq_length,
                    device=model.device,
                    sliding_window=sliding_window
                )
            ),
            total=iceildiv(
                iceildiv(len(data['val']), sliding_window),
                1
            )
    ):
        torch.cuda.empty_cache()
        with torch.no_grad():
            if x.shape[1] != seq_length:
                print("error sample")
                continue
            outputs = model(
                input_ids=x,
                labels=x,
                use_cache=use_cache)
        if idx < 10:
            print(f"the {idx} loss: ", outputs.loss.item())
        total_loss.append(outputs.loss.item())
        gc.collect()
        torch.cuda.empty_cache()
    mean_loss = sum(total_loss) / len(total_loss)
    ppl = 2.71828 ** mean_loss
    print("ppl", ppl)
    args.ppl = ppl
    import json
    with open("ppl.output.json", "a") as f:
        f.write(json.dumps(args.__dict__) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', default=8192, type=int)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--data_path', type=str, default="data/pg19.validation.bin")
    parser.add_argument('--scale', type=str, default="13b")
    parser.add_argument('--pretraining_length', type=int, default=4096)
    
    args = parser.parse_args()
    replace_with_chunkllama(args.pretraining_length, args.pretraining_length//4)
    model_path = f"meta-llama/llama-2-{args.scale}-hf"

    config = None

    if "70b" != args.scale:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", device_map="auto",
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16)
    data_path = args.data_path
    data = {'val': np.memmap(data_path, dtype=np.uint32, mode='r')}

    evaluate_ppl_all(seq_length=args.seq_len, sliding_window=256, args=args, model=model, data=data)

