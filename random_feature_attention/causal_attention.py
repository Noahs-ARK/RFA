""""Causal RFA."""

from typing import Dict, Optional

import math
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from random_feature_attention.rfa import RFA
from random_feature_attention.utils import upgrade_state_dict_named
from random_feature_attention.utils import random_project
from random_feature_attention.utils import normalize_attn_weights
from random_feature_attention.utils import EPS
from random_feature_attention.utils import build_random_matrices


def incremental_rfa(*,
                    phi_q: Tensor,
                    phi_k: Tensor,
                    v: Tensor,
                    s: Optional[Tensor] = None,
                    z: Optional[Tensor] = None) -> Tensor:
    """Loop causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        s: [bsz, num_heads, 2 * proj_dim, head_dim]
        z: [bsz, num_heads, 2 * proj_dim]
    """
    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)

    if s is None:
        assert z is None
        s = torch.zeros(
            (bsz, num_heads, proj_dim, head_dim),
            device=v.device, dtype=v.dtype)
        z = torch.zeros((bsz, num_heads, proj_dim), device=v.device, dtype=v.dtype)
    attns = []
    for i in range(tgt_len):
        s = s + torch.einsum("bhk,bhd->bhkd", phi_k[i, ...], v[i, ...])
        z = z + phi_k[i, ...]
        qs = torch.einsum("bhk,bhkd->bhd", phi_q[i, ...], s)
        qz = torch.einsum("bhk,bhk->bh", phi_q[i, ...], z)
        qz = qz.clamp_min(EPS)

        # [bsz, num_heads, head_dim]
        attns.append(qs / qz.unsqueeze(-1))
    # [tgt_len, bsz, num_heads, head_dim]
    attns = torch.stack(attns, dim=0).contiguous().view(
        tgt_len, bsz, num_heads * head_dim)
    return attns, s, z


def masked_rfa(*,
               phi_q: Tensor,
               phi_k: Tensor,
               v: Tensor,
               key_padding_mask: Optional[Tensor] = None,
               attn_mask: Optional[Tensor] = None) -> Tensor:
    """Masked causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        attn_mask (ByteTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None).
            [tgt_len, src_len]
    Return:
        attn: [tgt_len, bsz, num_heads * head_dim]
    """
    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)
    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.
    if key_padding_mask is not None and key_padding_mask.dim() == 0:
        key_padding_mask = None

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == tgt_len

    # [bsz, num_heads, tgt_len, src_len]
    attn_weights = torch.einsum("tbhk,sbhk->bhts", phi_q, phi_k)
    assert list(attn_weights.size()) == [bsz, num_heads, tgt_len, tgt_len]
    if key_padding_mask is not None:
        # [bsz, 1, 1, src_len]: bool
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        attn_weights = attn_weights.masked_fill(mask, 0.0)

    if attn_mask is not None:
        # [tgt_len, src_len]: bool. Masked are True
        mask = (attn_mask < 0.0)
        # [1, 1, tgt_len, src_len]
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(mask, 0.0)
    attn_weights = normalize_attn_weights(attn_weights, dtype=attn_weights.dtype)
    # [tgt_len, bsz, num_heads, head_dim]
    attn = torch.einsum("bhts,sbhd->tbhd", attn_weights, v)
    assert list(attn.size()) == [tgt_len, bsz, num_heads, head_dim]
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)
    return attn


def cuda_causal_rfa(*,
                    phi_q: Tensor,
                    phi_k: Tensor,
                    v: Tensor,
                    key_padding_mask: Optional[Tensor] = None
                    ) -> Tensor:
    """Cuda causal RFA implementation.

    Args:
        phi_q: [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_k: [tgt_len, bsz, num_heads, 2 * proj_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s.
    Return:
        attn: [tgt_len, bsz, num_heads * head_dim]
    """
    tgt_len, bsz, num_heads, proj_dim = phi_q.size()
    head_dim = v.size(-1)
    # This is part of a workaround to get around fork/join parallelism
    # not supporting Optional types.
    if key_padding_mask is not None and key_padding_mask.dim() == 0:
        key_padding_mask = None

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == tgt_len
        # [tgt_len, bsz]: bool
        mask = key_padding_mask.to(torch.bool).transpose(0, 1)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [tgt_len, bsz, 1, 1]
        phi_k = phi_k.masked_fill(mask, 0.0)

    phi_q = phi_q.contiguous().view(tgt_len, bsz * num_heads, -1)
    phi_k = phi_k.contiguous().view(tgt_len, bsz * num_heads, -1)
    v = v.contiguous().view(tgt_len, bsz * num_heads, head_dim)

    attn = RFA.apply(phi_q, phi_k, v)  # [tgt_len, bsz * num_heads, head_dim]
    attn = attn.contiguous().view(tgt_len, bsz, num_heads * head_dim)
    return attn


class CausalAttention(nn.Module):
    """Random feature cross attention."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        proj_dim: int,
        tau: float = 1.0,
        reparam_proj: bool = True,
        cuda_causal_rfa: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = proj_dim
        self.reparam_proj = reparam_proj
        self.cuda_causal_rfa = cuda_causal_rfa
        self.tau = tau
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if reparam_proj:
            self.sigma = Parameter(Tensor(num_heads, 1, head_dim))
        self.reset_parameters()
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self):
        gain = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)

        if self.reparam_proj:
            nn.init.constant_(self.sigma, 1.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        random_matrices: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        saved_state: Optional[Dict[str, Optional[Tensor]]] = None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            x: [tgt_len, bsz, embed_dim]
            random_matrices: [num_heads, proj_dim, head_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        assert list(x.size()) == [tgt_len, bsz, embed_dim]

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        k = k.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        v = v.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=self.tau,
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)
        phi_q = random_project(
            x=q,
            random_matrices=random_matrices
        )
        phi_k = random_project(
            x=k,
            random_matrices=random_matrices
        )
        if saved_state is not None:
            # Incremental decoding
            assert tgt_len == 1
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_s" in saved_state:
                assert "prev_s" in saved_state
                prev_s = saved_state["prev_s"]
                prev_z = saved_state["prev_z"]
                assert prev_s is not None
                assert prev_z is not None
            else:
                prev_s, prev_z = None, None
            # [tgt_len, bsz, embed_dim]
            attn, s, z = incremental_rfa(
                phi_q=phi_q, phi_k=phi_k, v=v, s=prev_s, z=prev_z)

            saved_state["prev_s"], saved_state["prev_z"] = s, z
        else:
            if self.cuda_causal_rfa:
                # [tgt_len, bsz, embed_dim]
                attn = cuda_causal_rfa(
                    phi_q=phi_q, phi_k=phi_k, v=v,
                    key_padding_mask=key_padding_mask)
            else:
                # [tgt_len, bsz, embed_dim]
                attn = masked_rfa(phi_q=phi_q, phi_k=phi_k, v=v,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask)

        # [tgt_len, bsz, embed_dim]
        attn = self.out_proj(attn)
        return attn
