import torch
from old_model import MoEUTLM
import profiles
vocab_size = 50320
batch_size = 16
sequence_len = 1024

model = MoEUTLM(vocab_size=vocab_size,d_model=128, n_layers=4, n_heads=4, ff_n_experts=10, att_n_experts=8, d_head=36, 
    ff_k=10, group_size=2, ).cuda()

tokens = torch.randint(0, vocab_size, (batch_size, sequence_len)).cuda()

model(tokens)