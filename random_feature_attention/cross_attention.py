""""Cross RFA."""

from typing import List, Optional, Tuple

import math
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from random_feature_attention.utils import CrossAttentionState
from random_feature_attention.utils import upgrade_state_dict_named
from random_feature_attention.utils import random_project
from random_feature_attention.utils import load_random_matrices
from random_feature_attention.utils import sample_random_matrices
from random_feature_attention.utils import EPS
from random_feature_attention.utils import build_random_matrices


class CrossAttentionProjectLayer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        proj_dim: int,
        tau: float = 1.0,
        reparam_proj: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = proj_dim
        self.reparam_proj = reparam_proj
        self.tau = tau
        assert self.num_heads * self.head_dim == self.embed_dim

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        if self.reparam_proj:
            self.sigma = Parameter(Tensor(self.num_heads, 1, self.head_dim))

        self.reset_parameters()
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        if self.reparam_proj:
            nn.init.constant_(self.sigma, 1.)

    def project_and_reshape(
        self,
        *,
        encoder_output: Tensor,
        random_matrices: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            random_matrices: num_layers * [num_heads, proj_dim, head_dim]
            mask: [src_len, bsz, 1, 1]: bool
        Return:
            s: [bsz, num_heads, proj_dim, head_dim]
            z: [bsz, num_heads, proj_dim]
        Einsum notations:
            b: bsz
            s: seq_len
            n: num_layers
            h: num_heads
            k: proj_dim
            d: head_dim
        """
        src_len, bsz, _ = encoder_output.size()

        # [src_len, bsz, num_head * head_dim]
        projected_k = self.k_proj(encoder_output)
        projected_v = self.v_proj(encoder_output)

        # [src_len, bsz, num_heads, head_dim]
        projected_k = projected_k.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)
        projected_v = projected_v.contiguous().view(
            src_len, bsz, self.num_heads, self.head_dim)
        random_matrices = build_random_matrices(
            random_matrices=random_matrices,
            tau=self.tau,
            sigma=self.sigma if self.reparam_proj else None,
            reparam_proj=self.reparam_proj)

        # [seq_len, bsz, num_heads, 2 * proj_dim]
        phi_k = random_project(
            x=projected_k,
            random_matrices=random_matrices
        )
        if mask is not None:
            # mask: [src_len, bsz, 1, 1]
            # phi_k: [src_len, bsz, num_heads, 2 * proj_dim]
            phi_k = phi_k.masked_fill(mask, 0.0)

        # [bsz, num_heads, proj_dim, head_dim]
        s = torch.einsum("sbhk,sbhd->bhkd", phi_k, projected_v)
        z = torch.sum(phi_k, dim=0)  # [bsz, num_heads, head_dim]

        return s, z, random_matrices, self.tau

    def forward(
        self,
        *,
        encoder_output: Tensor,
        random_matrices: Tensor,
        mask: Optional[Tensor] = None
    ) -> CrossAttentionState:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            random_matrices: [num_heads, proj_dim, head_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        seq_len, bsz, embed_dim = encoder_output.size()
        assert embed_dim == self.embed_dim

        s, z, random_matrices, tau = self.project_and_reshape(
            encoder_output=encoder_output,
            random_matrices=random_matrices,
            mask=mask
        )
        return CrossAttentionState(s, z, random_matrices, tau)


class CrossAttentionProject(nn.Module):
    """Encoder output projection for random feature cross attention."""
    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        proj_dim: int,
        tau: float = 1.0,
        reparam_proj: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = proj_dim
        assert self.num_heads * self.head_dim == self.embed_dim

        self.layers = [
            CrossAttentionProjectLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                proj_dim=proj_dim,
                tau=tau,
                reparam_proj=reparam_proj)
            for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)

        self.random_matrices = load_random_matrices(
            head_dim=self.head_dim,
            proj_dim=self.proj_dim,
            dtype=torch.float16)
        self.random_matrices_eval = sample_random_matrices(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            random_matrices=self.random_matrices,
            is_training=False)
        self.random_matrices_eval = nn.Parameter(
            self.random_matrices_eval, requires_grad=False)

    def forward(
        self,
        *,
        encoder_output: Tensor,
        key_padding_mask: Optional[Tensor] = None
    ) -> List[CrossAttentionState]:
        """
        Args:
            encoder_output: [seq_len, bsz, embed_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        src_len, bsz, embed_dim = encoder_output.size()
        assert embed_dim == self.embed_dim

        random_matrices = sample_random_matrices(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            random_matrices=self.random_matrices,
            is_training=True) if self.training else self.random_matrices_eval
        mask = key_padding_mask
        if mask is not None and mask.dim() == 0:
            mask = None
        if mask is not None:
            mask = mask.transpose(0, 1)
            assert mask.size(0) == src_len and mask.size(1) == bsz
            # [src_len, bsz, 1, 1]: bool
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        states = []
        for i in range(self.num_layers):
            states.append(self.layers[i](
                encoder_output=encoder_output,
                random_matrices=random_matrices[i] if self.training else random_matrices,
                mask=mask))
        return states


class CrossAttention(nn.Module):
    """Random feature cross attention."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        proj_dim: int,
        tau: float = 1.0,
        reparam_proj: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = proj_dim
        self.reparam_proj = reparam_proj
        self.tau = tau
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.reset_parameters()
        self.upgrade_state_dict_named = upgrade_state_dict_named

    def reset_parameters(self):
        gain = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)

    def rfa(
        self,
        query: Tensor,
        s: Tensor,
        z: Tensor,
        random_matrices: Tensor
    ) -> Tensor:
        # [tgt_len, bsz, num_heads, 2 * proj_dim]
        phi_q = random_project(
            x=query,
            random_matrices=random_matrices)

        qs = torch.einsum("tbhk,bhkd->tbhd", phi_q, s)
        qz = torch.einsum("tbhk,bhk->tbh", phi_q, z).abs().clamp_min(EPS)
        # [tgt_len, bsz, num_heads, head_dim]
        attn = qs / qz.unsqueeze(-1)
        return attn

    def forward(
        self,
        query: Tensor,
        state: CrossAttentionState
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            state: s, z, random_matrices
                s [bsz, num_heads, 2 * proj_dim, head_dim]
                z [bsz, num_heads, 2 * proj_dim]
                random_matrices: [num_heads, proj_dim, head_dim]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        s, z, random_matrices, tau = state
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, self.head_dim)
        attn = self.rfa(q, s, z, random_matrices)

        assert list(attn.size()) == [tgt_len, bsz, self.num_heads, self.head_dim]
        attn = attn.contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)
        # [tgt_len, bsz, embed_dim]
        attn = self.out_proj(attn)
        return attn
