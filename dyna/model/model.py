import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from dyna.model.cvmm import cvmm, cvmm_prepare_sel2, CVMMSel
from dataclasses import dataclass
from flash_attn.ops.triton.layer_norm import RMSNorm
from composer.models import HuggingFaceModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from omegaconf import DictConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from llmfoundry.utils.builders import build_metric
from torch.nn.attention import SDPBackend, sdpa_kernel
CROSS_ENTROPY_IGNORE_INDEX = -100
DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    "language_cross_entropy",
    "language_perplexity",
    "token_accuracy",
]


def get_targets(labels: torch.Tensor) -> torch.Tensor:
    targets = torch.roll(labels, shifts=-1)
    targets[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
    return targets


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    shift_labels: bool,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    targets = get_targets(labels) if shift_labels else labels

    losses = loss_fn(
        outputs.logits.view(-1, outputs.logits.size(-1)),  # type: ignore
        targets.view(-1),
    )

    if torch.all(targets == loss_fn.ignore_index):  # type: ignore
        loss = losses.sum()
    else:
        loss = losses.sum() / (targets != loss_fn.ignore_index).sum()  # type: ignore

    return loss


class MoEUTConfig(PretrainedConfig):
    model_type = "moeut"

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_ffn_experts: int,
        n_att_experts: int,
        d_head: Optional[int] = None,
        group_size: int = 2,
        ff_k: int = 8,
        att_k: int = 2,
        ff_expert_dropout: float = 0.0,
        att_expert_dropout: float = 0.0,
        ff_expert_size: int = 128,
        dropout: float = 0.0,
        entropy_reg: float = 0.01,
        att_entropy_reg: float = 0.001,
        shift_labels: bool = True,
        scale_add: bool = True,
        prot_emb: bool = False,
        **kwargs,
    ):
        super().__init__(**{"model_type": self.model_type})
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_ffn_experts = n_ffn_experts
        self.n_att_experts = n_att_experts
        self.d_head = d_head
        self.group_size = group_size
        self.ff_k = ff_k
        self.att_k = att_k
        self.ff_expert_dropout = ff_expert_dropout
        self.att_expert_dropout = att_expert_dropout
        self.ff_expert_size = ff_expert_size
        self.dropout = dropout
        self.entropy_reg = entropy_reg
        self.att_entropy_reg = att_entropy_reg
        self.shift_labels = shift_labels
        self.scale_add = scale_add
        self.prot_emb = prot_emb
        # Accept and store any additional keys for forward compatibility
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]


def log_mean(x: torch.Tensor, dim: int = 0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return -(l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


KVCache = Optional[Dict[str, torch.Tensor]]
MultilayerKVCache = Optional[Dict[int, KVCache]]


@dataclass
class MoEUTOutput:
    outputs: torch.Tensor
    reg_loss: torch.Tensor
    cache: MultilayerKVCache


class SigmaMoE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_size: int,
        k: int,
        activation=F.relu,
        v_dim: Optional[int] = None,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.k_dim = d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation = activation
        self.expert_dropout = expert_dropout

        self.sel_hist = []

        self.keys = torch.nn.Parameter(
            torch.empty(self.n_experts, self.k_vec_dim, self.expert_size)
        )
        self.values = torch.nn.Parameter(
            torch.empty(self.n_experts, self.expert_size, self.v_dim)
        )
        self.expert_sel = torch.nn.Parameter(
            torch.empty(self.n_experts, self.k_vec_dim)
        )

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(
            self.values, 0, std_scale / math.sqrt(self.n_experts * self.expert_size)
        )

        self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    # def get_reg_loss(self) -> torch.Tensor:
    #     if not self.sel_hist:
    #         return 0

    #     # Average over time and layers.
    #     loss = entropy_reg(torch.stack(self.sel_hist, dim=-2).flatten(-3, -2), -2)
    #     self.sel_hist = []
    #     return loss

    def forward(
        self, input: torch.Tensor, sel_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Selection score calculation
        sel = F.linear(
            sel_input if sel_input is not None else input, self.expert_sel, None
        )
        if self.training:
            self.sel_hist.append(sel)

        # Selection activation and topk
        sel = F.sigmoid(sel)

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))

        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = cvmm_prepare_sel2(sel_index.int())

        # "Up-projection" layer for each head
        scores = cvmm(input, sel_indices, self.keys)
        scores = self.activation(scores)

        # Down projection layer for each head
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        out = cvmm(scores, sel_indices, self.values)

        res = out.view(*input.shape[:-1], self.v_dim)
        return res


class SwitchHeadCore(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        dropout: float = 0.0,
        d_head: Optional[int] = None,
        expert_dropout: float = 0.0,
        moe_k: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.pe_size = self.d_model
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        self.n_experts = n_experts

        self.sel_hist = []

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.d_head = d_head or (d_model // n_heads)

        self.q = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model, self.d_head)
            )
            self.o = torch.nn.Parameter(
                torch.empty(
                    self.n_heads * self.n_experts,
                    self.d_head,
                    self.d_model,
                )
            )
            self.sel_v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model)
            )
        else:
            self.v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.d_head, self.d_model)
            )
            self.o = torch.nn.Parameter(
                torch.empty(self.d_model, self.n_heads * self.d_head)
            )

        self.sel_o = torch.nn.Parameter(
            torch.empty(self.n_heads * self.n_experts, self.d_model)
        )

        self.register_buffer(
            "scale",
            torch.full([1], 1.0 / math.sqrt(self.d_head)),
            persistent=False,
        )

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, std_scale / math.sqrt(self.d_model))
            self.renorm_rows(self.sel_v)

        torch.nn.init.normal_(self.sel_o, 0, std_scale / math.sqrt(self.d_model))
        self.renorm_rows(self.sel_o)

        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(
            self.o, 0, std_scale / math.sqrt(self.n_heads * self.d_head)
        )

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_mask_tensor(
        self, src_len: int, mask: Optional[AttentionMask]
    ) -> Optional[torch.Tensor]:
        if mask is None or (mask[0] is None and mask[1] is None):
            return None

        # mask[0]: [..., N_out, N_in]
        # mask[1]: [B, ...., N_in]
        # True where it has to be masked

        if mask[0] is not None:
            n_pad = src_len - mask[0].shape[-1]
            if n_pad > 0:
                pm = F.pad(mask[0], (n_pad, 0), "constant", value=False)
            else:
                pm = mask[0]

        if mask[0] is None:
            m = mask[1].unsqueeze(-2).unsqueeze(-2)
        elif mask[1] is None:
            m = pm
        else:
            m = mask[1].unsqueeze(-2).unsqueeze(-2) | pm

        return m

    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> Tuple[CVMMSel, torch.Tensor]:

        # t :  batch_size, seq_len, d_model
        # w : n_heads * n_experts, d_model
        # print("t", t.shape)
        # print("w", w.shape)

        sel = F.linear(t, w).float()
        # sel : batch_size, seq_len, n_heads * n_experts
        sel = sel_raw = sel.view(*sel.shape[:-1], self.n_heads, -1)
        # sel : batch_size, seq_len, n_heads, n_experts

        sel = sel.sigmoid()
        # sel : batch_size, seq_len, n_heads, n_experts
        # print("sel", sel.shape)

        # C This tells which token goes to which expert -- here is where we can skip ?
        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float("-inf"))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)
        # sel_index: batch_size, seq_len, n_heads, attn_n_experts
        # sel_val : batch_size, seq_len, n_heads, attn_n_experts
        # sel_val is the per token expert which was assembled

        # We have the two experts selected

        # This is like a positional encoding, but for the experts?
        sel_index_shifted = (
            torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype)
            * self.n_experts
        ).unsqueeze(-1) + sel_index
        # sel_index_shifted : batch_size, seq_len, n_heads, attn_n_experts

        # #print("sel_index_shifted", sel_index_shifted.shape)
        # #print((torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1))

        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2, -1), sel_val), sel_raw

    def attend(
        self,
        pos_offset: int,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(
        self,
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        mask: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache = None,
        pos_offset: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, KVCache]:

        # pos_offset = q_src.shape[1] - k_src.shape[1]
        # assert pos_offset >= 0

        scale = self.scale.sqrt()

        q = self.q(q_src)
        k = self.k(k_src)
        # q : batch_size, seq_len, d_head * n_heads
        # k : batch_size, seq_len, d_head * n_heads
        q = q * scale.type_as(q)
        k = k * scale.type_as(k)

        # Wsv, Wso matrices?
        # Lets
        # Values and output expert selection
        if self.n_experts > 1:
            v_sel, v_sel_raw = self.get_sel(k_src, self.sel_v)
            o_sel, o_sel_raw = self.get_sel(q_src, self.sel_o)
            if self.training:
                self.sel_hist.append((o_sel_raw, v_sel_raw))

            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

        if kv_cache is not None:
            v = torch.cat([kv_cache["v"], v], dim=-2) if "v" in kv_cache else v
            k = torch.cat([kv_cache["k"], k], dim=-2) if "k" in kv_cache else k
            kv_cache = {"v": v, "k": k}

        q = self.dropout(q)
        # print("qshape")
        # print(q.shape)
        # print(k.shape)
        # print(self.n_experts)
        res = self.attend(
            pos_offset, positions, v, k, q, self.get_mask_tensor(v.shape[-2], mask)
        )
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            # The output selection indices are calculated from the current state and are also used for projecting "q".
            # But that projection needs to create multiple copies for the different heads. Here we already have the
            # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
            # weight. We also want to compute not only the weighted average between the top-k elements, but also
            # of the different heads. So reshape the reduction weight accordingly.
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out, kv_cache


class RotaryPosEncoding(torch.nn.Module):
    # RoPE based on: https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = torch.tensor(seq_dim)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rot(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:

        # the offset here is from zero so we will ideally make it mask
        # remove the masked elements
        if positions is None:
            sin, cos = self.get(x)
        else:
            sin, cos = self.get1(positions, x)
        sin = sin.narrow(self.seq_dim, offset, x.shape[self.seq_dim])
        cos = cos.narrow(self.seq_dim, offset, x.shape[self.seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def get(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_dim], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            self.cos_cached = emb.cos().view(*tgt_shape)
            self.sin_cached = emb.sin().view(*tgt_shape)

        return self.sin_cached, self.cos_cached

    def get1(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("ki,j->kij", x, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

        tgt_shape = [1] * q.ndim
        tgt_shape[0] = q.shape[0]
        tgt_shape[1] = 1
        tgt_shape[self.seq_dim] = q.shape[self.seq_dim]
        tgt_shape[-1] = q.shape[-1]

        return emb.sin().view(*tgt_shape), emb.cos().view(*tgt_shape)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0, positions=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, pos_offset, positions), self.apply_rot(k, 0, None)


class SwitchHeadRope(SwitchHeadCore):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        dropout: float = 0.0,
        d_head: int | None = None,
        expert_dropout: float = 0.0,
        moe_k: int = 2,
        rotate_fraction: float = 0.5,
        rope_base: float = 10000,
    ):
        super().__init__(
            d_model, n_heads, n_experts, dropout, d_head, expert_dropout, moe_k
        )
        self.n_rotate = int(rotate_fraction * self.d_head)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def rotate(
        self, q: torch.Tensor, k: torch.Tensor, offset: int, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.d_head:
            r_k = k[..., : self.n_rotate]
            nr_k = k[..., self.n_rotate :]
            r_q = q[..., : self.n_rotate]
            nr_q = q[..., self.n_rotate :]

            r_q, r_k = self.pe(r_q, r_k, torch.tensor(offset), positions)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.pe(q, k, torch.tensor(offset), positions)

    def attend(
        self,
        pos_offset: int,
        positions: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0, positions)

        # Masking is missing -- needs to be assembled
        # Maksing problem confirmed!
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, scale=1.0)


#  d_model: int,
#         n_heads: int,
#         ff_expert_size: int,
#         ff_n_experts: int,
#         att_n_experts: int,
#         d_head: Optional[int] = None,
#         att_k: int = 2,
#         ff_k: int = 8,
#         ff_expert_dropout: float = 0.0,
#         att_expert_dropout: float = 0.0,
#         dropout: float = 0.0,
#         attention=SwitchHeadRope,
class MoEUTLayer(torch.nn.Module):
    def __init__(self, config: MoEUTConfig):
        super().__init__()
        # Attention MoE
        self.attention = SwitchHeadRope(
            config.d_model,
            config.n_heads,
            config.n_att_experts,
            d_head=config.d_head,
            moe_k=config.att_k,
            expert_dropout=config.att_expert_dropout,
        )
        # FFN MoE
        self.ffn = SigmaMoE(
            config.d_model,
            config.n_ffn_experts,
            config.ff_expert_size,
            k=config.ff_k,
            expert_dropout=config.ff_expert_dropout,
        )
        self.attn_pre = RMSNorm(config.d_model)
        self.attn_post = RMSNorm(config.d_model)
        self.ffn_pre = RMSNorm(config.d_model)
        self.ffn_post = RMSNorm(config.d_model)

        self.drop = torch.nn.Dropout(config.dropout)
        self.scale_add = config.scale_add
        self.group_size = config.group_size

    def forward(
        self,
        x: torch.Tensor,
        layer_index: int,
        e: torch.Tensor | None = None,
        router: torch.nn.Parameter | None = None,
        cum_sum: torch.Tensor | None = None,
        mask: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: KVCache = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        condition = True
        layer_index = self.group_size * layer_index + 1

        # Calc s exit
        s_exit = F.sigmoid(F.linear(x, router))
        # add it to sum
        cum_sum += s_exit

        # find the original pos offset
        pos_offset = 0

        # make a mask
        skip_mask = cum_sum < 0.5
        if torch.all(skip_mask == False):
            return x, kv_cache, False
        lengths = skip_mask.sum(dim=1)  # number of valid entries per batch
        max_len = lengths.max()
        positions = torch.zeros(x.shape[0], max_len, dtype=torch.long, device=x.device)

        # compact batch
        out = torch.zeros(x.shape[0], max_len, x.shape[-1], device=x.device)

        xnorm = self.attn_pre(x)
        # Pack the batch to be something smaller - VECTORIZED VERSION
        # Get all valid positions in one go
        batch_idx, seq_idx = skip_mask.nonzero(as_tuple=True)
        
        # Create a mapping for efficient packing
        if batch_idx.numel() > 0:
            # Count valid tokens per batch
            batch_counts = torch.bincount(batch_idx, minlength=x.shape[0])
            cumsum_counts = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]), dim=0)
            
            # Create output indices
            local_indices = torch.arange(batch_idx.numel(), device=x.device) - cumsum_counts[batch_idx]
            valid_mask = local_indices < max_len
            
            if valid_mask.any():
                # Pack efficiently
                out[batch_idx[valid_mask], local_indices[valid_mask]] = xnorm[batch_idx[valid_mask], seq_idx[valid_mask]]
                positions[batch_idx[valid_mask], local_indices[valid_mask]] = seq_idx[valid_mask]

        # Give the batch and location info to the attention mechanism
        # There skip the offset assertion by providing a location into and offset precalculated
        # Then, apply the rope is a sparse way
        # Then, attend
        # Then return to the original dimension using:
        # processed_flat = torch.cat([
        #     processed[i, :lengths[i]]
        #     for i in range(B)
        # ], dim=0)  # shape (total_valid, D)

        # # Prepare output
        # restored = torch.zeros_like(a)
        # restored[b] = processed_flat

        att, kv_cache = self.attention(
            out,
            xnorm,
            xnorm,
            mask,
            kv_cache=kv_cache,
            pos_offset=pos_offset,
            positions=positions,
        )

        # att shape batch_size x seq_len x d_model
        if self.scale_add:
            if e is not None:
                # If we scale addition, we have to scale the update
                # Otherwise, we just add the update
                # Do this element wise, so layer index will now be a tensor as well
                x = x - e

                # This will be interesting
                # The RMS norm will maybe exxagerate the effects of single tokens
                attn = self.drop(self.attn_post(att))
                # More efficient: use boolean indexing instead of concatenation
                x[skip_mask] = layer_index / (layer_index + 1) * x[
                    skip_mask
                ] + attn.view(-1, attn.shape[-1])[:skip_mask.sum()] * cum_sum[skip_mask].unsqueeze(1) / (layer_index + 1)

                x = x + e
            else:
                attn = self.drop(self.attn_post(att))
                # More efficient: use boolean indexing instead of concatenation
                x[skip_mask] = layer_index / (layer_index + 1) * x[
                    skip_mask
                ] + attn.view(-1, attn.shape[-1])[:skip_mask.sum()] * cum_sum[skip_mask].unsqueeze(1) / (layer_index + 1)

        else:
            x = self.drop(self.attn_post(att)) + x

        layer_index += 1

        # here we prenorm
        xnorm = self.ffn_pre(x)
        # Vectorized packing for FFN (reuse the same logic)
        batch_idx, seq_idx = skip_mask.nonzero(as_tuple=True)
        
        if batch_idx.numel() > 0:
            batch_counts = torch.bincount(batch_idx, minlength=x.shape[0])
            cumsum_counts = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]), dim=0)
            local_indices = torch.arange(batch_idx.numel(), device=x.device) - cumsum_counts[batch_idx]
            valid_mask = local_indices < max_len
            
            if valid_mask.any():
                out[batch_idx[valid_mask], local_indices[valid_mask]] = xnorm[batch_idx[valid_mask], seq_idx[valid_mask]]
                positions[batch_idx[valid_mask], local_indices[valid_mask]] = seq_idx[valid_mask]
        upd = self.ffn(out, out)
        # upd shape batch_size x seq_len x d_model
        if self.scale_add:
            # If we scale addition, we have to scale the update
            # Otherwise, we just add the update
            if e is not None:

                x = x - e

                # This will be interesting
                # The RMS norm will maybe exxagerate the effects of single tokens
                ffn = self.drop(self.ffn_post(upd))
                # More efficient: use boolean indexing instead of concatenation
                x[skip_mask] = layer_index / (layer_index + 1) * x[
                    skip_mask
                ] + ffn.view(-1, ffn.shape[-1])[:skip_mask.sum()] * cum_sum[skip_mask].unsqueeze(1) / (layer_index + 1)

                x = x + e
            else:
                x = layer_index / (layer_index + 1) * x + self.drop(
                    self.ffn_post(upd)
                ) / (layer_index + 1)
        else:
            # If we do not scale addition, we just add the update
            x = self.drop(self.ffn_post(upd)) + x

        return x, kv_cache, True


class MoEUTPretrainedModel(PreTrainedModel):
    config_class = MoEUTConfig
    base_model_prefix: str = "mouet"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None

    # - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
    # - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
    # - **path** (`str`) -- A path to the TensorFlow checkpoint.


class MoEUT(MoEUTPretrainedModel):
    # Two major things here
    # First, get the computation graph from the preserving e
    # second, do the early exiting
    def __init__(
        self,
        config: MoEUTConfig,
    ):
        super().__init__(config)
        self.entropy_reg = config.entropy_reg
        self.att_entropy_reg = config.att_entropy_reg
        self.group_size = config.group_size
        self.n_repeats = 4
        self.d_model = config.d_model
        self.layers = torch.nn.ModuleList(
            [MoEUTLayer(config) for _ in range(config.group_size)]
        )
        self.router = torch.nn.Parameter(torch.empty(self.d_model))
        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor | None = None,
        mask: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: MultilayerKVCache = None,
    ) -> MoEUTOutput:
        # x input of shape batch_size x seq_len x d_model
        new_cache = {}
        condition = True
        cum_sum = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        # populate the kv cache based on absolute index of layer
        # Execute the layer
        layer_index_abs = 0
        while condition:
            # We are routing intra group
            # Intra means within
            # Inter means between
            for layer in self.layers:
                cache = (
                    kv_cache.get(layer_index_abs, {}) if kv_cache is not None else None
                )

                # execute the layer
                # x shape batch_size x seq_len x d_model
                x, new_cache[layer_index_abs], condition = layer(
                    x, layer_index_abs, e, self.router, cum_sum, mask, kv_cache=cache
                )
                if not condition:
                    break
                # x shape batch_size x seq_len x d_model
                layer_index_abs += 1
            if layer_index_abs > 12:
                break

        # Collect regularizaiton losses. Must be at the end because it is across the layers.
        # reg_loss = torch.zeros(1, device=x.device, dtype=torch.float32)
        # for layer in self.modules():
        #     if isinstance(layer, SigmaMoE):
        #         reg_loss = reg_loss + self.entropy_reg * layer.get_reg_loss()
        #     elif isinstance(layer, SwitchHeadCore):
        #         reg_loss = reg_loss + self.att_entropy_reg * layer.get_reg_loss()

        return CausalLMOutputWithPast(0, x, new_cache if kv_cache is not None else None)

    @torch.no_grad
    def reset_parameters(self):
        scale = math.sqrt(2 / (self.n_repeats * len(self.layers)))
        torch.nn.init.normal_(self.router, 0, scale / math.sqrt(self.d_model))
        for layer in self.modules():
            if isinstance(layer, (SwitchHeadCore, SigmaMoE)):
                layer.reset_parameters(scale)
            elif isinstance(layer, RMSNorm):
                layer.reset_parameters()


#  vocab_size: int,
#         d_model: int,
#         n_layers: int,
#         n_heads: int,
#         ff_n_experts: int,
#         att_n_experts: int,
#         d_head: Optional[int] = None,
#         group_size: int = 2,
#         ff_k: int = 8,
#         att_k: int = 2,
#         ff_expert_dropout: float = 0.0,
#         att_expert_dropout: float = 0.0,
#         ff_expert_size: int = 128,
#         dropout: float = 0.0,
#         entropy_reg: float = 0.01,
#         att_entropy_reg: float = 0.001,
#         attention=SwitchHeadRope,
class MoEUTLM(MoEUTPretrainedModel):
    def __init__(
        self,
        config: MoEUTConfig,
    ):
        super().__init__(config)
        self.transformer = MoEUT(config)
        self.n_layers = config.n_layers
        self.prot_emb = config.prot_emb
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size)
        self.out_norm = RMSNorm(config.d_model)
        self.reset_parameters()

    @torch.no_grad
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(
            self.embedding.weight, mode="fan_in", nonlinearity="linear"
        )
        self.transformer.reset_parameters()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        src_len_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:

        # Check that we have either input_ids or inputs_embeds
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds, not both"
            )

        # x input of shape batch_size x seq_len -- tokens so ints
        # kv_cache: feed an empty dict to start caching
        kv_cache = {}
        if attention_mask is None:
            attention_mask = self.generate_causal_attention_mask(input_ids.shape[-1])

        if input_ids is not None:
            x = self.embedding(input_ids)

        # x shape batch_size x seq_len x d_model
        e = None
        if self.prot_emb:
            e = x.clone()
        out = self.transformer(x, e, (attention_mask, src_len_mask), kv_cache)
        # print(x.shape)
        # x shape batch_size x seq_len x d_model
        out.logits = self.lm_head(self.out_norm(out.logits))
        # out.outputs shape batch_size x seq_len x vocab_size
        return out

    def generate_causal_attention_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=self.lm_head.weight.device),
            diagonal=1,
        )

class ComposerMoEUT(HuggingFaceModel):
    def __init__(
        self,
        config: MoEUTConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        # train_metric_names: list | None = None,
        # loss_fn=None,
    ):
        model = self.model_class(config)
        self.vocab_size = config.vocab_size
        eval_metrics = {}
        self.shift_labels = config.shift_labels
        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS
        train_metrics = [build_metric(metric, {}) for metric in train_metric_names]
        super().__init__(
            model=model,
            tokenizer=tokenizer,  # type: ignore
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=config.shift_labels,
            allow_embedding_resizing=True,
        )

    @property
    def model_class(self):
        return MoEUTLM

    def forward(self, batch) -> CausalLMOutputWithPast:
        return self.model(
            input_ids=batch.get("input_ids", None),
            inputs_embeds=batch.get("inputs_embeds", None),
            attention_mask=batch.get("attention_mask", None),
        )

    def loss(self, outputs: CausalLMOutputWithPast, batch) -> dict | torch.Tensor:

        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
            reduction="mean",
        )

        loss = compute_loss_from_logits(
            outputs,
            self.shift_labels,
            batch["input_ids"] if "input_ids" in batch else batch["labels"],
            loss_fn,
        )
        return loss
