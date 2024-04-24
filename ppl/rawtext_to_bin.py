from transformers import AutoTokenizer
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="llama3")
args = parser.parse_args()

model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
content = open("pg19_raw.txt").read()
if args.model == "mistral":
    batch_size = 100000

    # max_length = 1024
    batches = [
        content[i : i + batch_size] for i in range(0, len(content), batch_size)
    ]
    tokenized_batches = []
    with tqdm(total=len(batches)) as pbar:
        for batch in batches:
            tokenized = tokenizer(
                batch,
                return_tensors="np",
            )
            tmp = tokenized["input_ids"][0][1:]
            print(tmp)
            tokenized_batches.append(tmp)
            pbar.update(1)

    ids = np.concatenate(tokenized_batches, axis=-1)
else:
    ids = tokenizer.encode(content)
print(len(ids))
print("begin write")
p = np.memmap(f"./pg19_{args.model}.validation.bin", dtype=np.uint32, mode='write', shape=(len(ids)))
p[:] = ids[:]
