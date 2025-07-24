
import time
import torch


bs = 16
seq_len = 1024
d_model = 1024
q = torch.randn(bs, seq_len, d_model)
k = torch.randn(bs, seq_len, d_model)
v = torch.randn(bs, seq_len, d_model)
q2 = torch.randn(bs, seq_len, d_model)



def attention(q, k, v):
    t1 = time.time()
    
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # standard approach
    output = torch.matmul(attn_weights, v)
    t2 = time.time()
    print(f"Standard attention time: {t2 - t1:.6f}s")
    
    avg_time_zeroing = 0.0
    avg_time_masking = 0.0
    # Testing zeroing vs masking
    
    for i in range(0,1024,10):
        t1 = time.time()
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        indeces = torch.randint(0, seq_len, (bs, i))  # Randomly select 10 indices to zero out
        
        # Zeroing approach
        attn_weights[:, indeces, :] = 0.0
        output = torch.matmul(attn_weights, v)

        t2 = time.time()
        d_k = q.size(-1)
        mask = torch.ones_like(q, dtype=torch.bool)
        mask[:, indeces, :] = False
        q1 = q[mask].view(bs, -1, d_model)
        
        scores = torch.matmul(q1, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        indeces = torch.randint(0, seq_len, (bs, i))  # Randomly select 10 indices to zero out
        # masking approach

        # attn_weights = attn_weights[mask].view(bs, -1, seq_len)
        output = torch.matmul(attn_weights, v)
        # index_expanded = indeces.unsqueeze(-1).expand(-1, -1, d_model)
        # print(index_expanded.shape, output.shape)
        # q2.scatter_add_(dim=1, index=index_expanded, src=output)
        t3 = time.time()
        avg_time_zeroing += (t2 - t1)
        avg_time_masking += (t3 - t2)
        print(f"Iteration {i}: Zeroing time: {t2 - t1:.6f}s, Masking time: {t3 - t2:.6f}s")
    print(f"Average Zeroing time: {avg_time_zeroing / (1024 // 10):.6f}s")
    print(f"Average Masking time: {avg_time_masking / (1024 // 10):.6f}s")
    
    indeces = torch.randint(0, seq_len, (bs, 100)) 
    mask = torch.ones_like(q, dtype=torch.bool)
    mask[:, indeces, :] = False
    q1 = q[mask].view(bs, -1, d_model)
    attn_w1 = torch.matmul(q1, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_w2 = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn_w2 = attn_w2[mask].view(bs, -1, d_model)
    print(torch.allclose(attn_w1, attn_w2))
    return output

output = attention(q, k, v)
