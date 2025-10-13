from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor | None
    out_index: torch.Tensor | None
    reduction_weight: torch.Tensor | None = None

    def clone(self) -> CVMMSel:
        return CVMMSel(
            self.raw_sel,
            self.sel,
            self.sel_index,
            self.out_index,
            self.reduction_weight,
        )


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()
    return CVMMSel(sel, ssel.view_as(sel), sel_index, None)


def cvmm_prepare_sel2(sel: torch.Tensor, w: torch.Tensor | None = None) -> CVMMSel:
    # Has multiple selections for each batch element
    n_per_batch = sel.shape[-1]

    # indices = torch.arange(sel.nelement() // n_per_batch, device=sel.device, dtype=torch.int32)
    # indices = indices.repeat_interleave(n_per_batch).flatten()

    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()

    # in_index = indices[sel_index]
    in_index = sel_index // n_per_batch

    return CVMMSel(sel, ssel.view_as(sel), in_index, sel_index, w)
