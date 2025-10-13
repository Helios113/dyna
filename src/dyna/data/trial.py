import os

from streaming import StreamingDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")

os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"

# Remote path where full dataset is persistently stored
remote = "s3://loop-llm/smoll_corpus/fineweb-edu-dedup/train"

# Local working dir where dataset is cached during operation
local = "/nfs-share/pa511/code_bases/abbie/data_cache"

# Create streaming dataset
dataset = StreamingDataset(local=local, remote=remote, shuffle=True, batch_size=16)

# Let's see what is in sample #1337...
sample = dataset[1337]
tokenized = tokenizer.encode(sample["text"], return_tensors="pt")
print(f"Sample 1337: {sample}")
print(f"Tokenized Sample 1337: {tokenized}")

# Create PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=16)

# for i in dataloader:
#     print(f"Batch {i}: {i}")
#     tokenized = tokenizer.encode(i['text'], return_tensors='pt')
#     print(f"Tokenized Batch {i}: {tokenized}")
#     break  # Just show the first batch for brevity
