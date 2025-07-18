import torch
import torch.nn.functional as F
from model import MoEUTLM
from old_model import MoEUTLM_old
import profiles
import time
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from composer.models import ComposerModel

batch_size = 8
context_window = 1024
vocab_size = 8000

tokens = torch.randint(0, vocab_size, (batch_size, context_window)).cuda()
old_model = MoEUTLM_old(vocab_size, **profiles.MoEUT_16M).cuda()
# old_model1 = MoEUTLM_old(vocab_size, **profiles.MoEUT_16M).cuda()
model = MoEUTLM_old(vocab_size, **profiles.MoEUT_16M).cuda()

print(tokens.shape)

def compare_models(model1, model2, tokens):
    model1.eval()
    model2.eval()
    for _ in range(10):
        out1 = model1(tokens)
        out2 = model2(tokens)
        
    
    n_runs = 100
    model1_times = []
    model2_times = []
    output_diffs = []

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_runs)):
            torch.cuda.synchronize()
            t1 = time.time()
            out1 = model1(tokens).outputs
            torch.cuda.synchronize()
            t2 = time.time()
            out2 = model2(tokens).outputs
            torch.cuda.synchronize()
            t3 = time.time()
            model1_times.append(t2 - t1)
            model2_times.append(t3 - t2)
            # Flatten outputs and compute difference
            diff = (torch.norm(out1.flatten() - out2.flatten())).cpu().numpy()
            output_diffs.append(diff)

    # Time stats
    model1_mean = np.mean(model1_times)
    model2_mean = np.mean(model2_times)
    print(f"model1 avg time {model1_mean}")
    print(f"model2 avg time {model2_mean}")

    # Plot time distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(model1_times, bins=20, alpha=0.7, label='Model 1')
    plt.hist(model2_times, bins=20, alpha=0.7, label='Model 2')
    plt.title('Inference Time Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    print(f"plotting time distributions: {model1_mean:.4f}s vs {model2_mean:.4f}s")

    # Plot output difference distribution
    plt.subplot(1, 2, 2)
    plt.hist(output_diffs, bins=50, alpha=0.7, color='purple')
    plt.title('Output Difference Distribution')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    

# Example usage:
# compare_models(old_model, old_model1, tokens)
compare_models(model, old_model, tokens)


