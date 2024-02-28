# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import random
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunkllama_attn_replace import replace_with_chunkllama


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

#### Setting for Llama2
B_INST, E_INST = "[INST]", "[/INST]"
BOS, EOS = "<s>", "</s>"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=16384,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    pretraining_length: int = field(
        default=4096,
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def rank0_write(*args):
    if local_rank == 0:
        with open("example.txt", "w") as f:
            f.write(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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


def write_jsonl(data, fn):
    with open(fn, "w") as f:
        for line in data:
            print(json.dumps(line), file=f)


def preprocess_shrotprompt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conversations = []
    for source in sources:
        inputs = source["inputs"]
        outputs = source["outputs"]
        whole_sequence = f"{BOS}{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {inputs[0]} {E_INST} {outputs[0]} {EOS}"
        if len(inputs) > 1 and len(outputs) > 1 and len(inputs)==len(outputs):
            for i in range(1, len(inputs)):
                whole_sequence += f"{BOS}{B_INST} {inputs[i]} {E_INST} {outputs[i]} {EOS}"
        conversations.append(whole_sequence)
    # Tokenize conversations
    rank0_write(conversations[0])

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    sep = E_INST
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(EOS)
        cur_len = 1
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(
                    f"WARNING: tokenization mismatch " f"{cur_len} vs. {total_len}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        # load data
        list_data_dict = load_dataset("L4NLP/chunkllama-long-dialog", split="train")
        random.shuffle(list_data_dict)
        rank0_print(len(list_data_dict))

        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess_shrotprompt(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        # load data
        list_data_dict = load_dataset("L4NLP/chunkllama-long-dialog", split="train")
        random.shuffle(list_data_dict)
        print("Num of training samples: ",len(list_data_dict))

        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        print(len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess_shrotprompt(sources, self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                attention_mask=data_dict["attention_mask"][0],
            )
        return data_dict


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_with_chunkllama(training_args.pretraining_length)

    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
