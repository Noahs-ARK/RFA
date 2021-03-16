"""RFA Utils."""
from typing import List, Optional, Tuple
import numpy as np
import torch
import os
from torch import Tensor


EPS = 1.0
SCALE = 0.1
RANDOM_MATRICES_PATH = os.path.join(os.path.dirname(__file__), '../random_matrices')


class CrossAttentionState(NamedTuple):
    s: Tensor
    z: Tensor
    random_matrices: List[Tensor]
    tau: float

def get_rfa_attr(args):
    args.use_linfa = getattr(args, "use_rfa", False)
    args.cross_proj_dim = getattr(args, "cross_proj_dim", 64)
    args.causal_proj_dim = getattr(args, "causal_proj_dim", 64)
    args.cross_tau = getattr(args, "cross_tau", 1.0)
    args.causal_tau = getattr(args, "causal_tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", True)
    return args


def add_rfa_args(parser):
    parser.add_argument('--use-rfa', action='store_true',
                        help='whether or not to use rfa')
    parser.add_argument('--cross-proj-dim', type=int, metavar='N',
                        help='projection size for cross rfa')
    parser.add_argument('--causal-proj-dim', type=int, metavar='N',
                        help='projection size for causal rfa')
    parser.add_argument('--cross-tau', type=float, metavar='D',
                        help='tau for rfa')
    parser.add_argument('--causal-tau', type=float, metavar='D',
                        help='tau for rfa')
    parser.add_argument('--reparam-proj', action='store_true',
                        help='whether or not to reparameterze random matrices in rfa')
    parser.add_argument('--cuda-causal-rfa', action='store_true',
                        help='whether or not to use custom cuda kernel for causal rfa')
    return parser


def load_random_matrices(
        *,
        head_dim: int,
        proj_dim: int,
        dtype: torch.dtype = torch.half) -> Tensor:

    # [num_random_matrices, proj_dim, head_dim]
    random_matrices = np.load(
        f"{RANDOM_MATRICES_PATH}/{head_dim}_{proj_dim}.npy")
    return torch.nn.Parameter(
        torch.tensor(random_matrices, dtype=dtype), requires_grad=False)


def sample_random_matrices(
        *,
        num_layers: int,
        num_heads: int,
        random_matrices: Tensor,
        is_training: bool = True):
    # random_matrices
    # [num_random_matrices, proj_dim, head_dim]

    if is_training:
        num_random_matrices = random_matrices.size(0)
        indices = np.random.choice(
            num_random_matrices,
            size=num_layers * num_heads,
            replace=False)
        # [num_layers * num_heads, proj_dim, head_dim]
        random_matrices = random_matrices[indices]
        sampled_random_matrices = []
        for i in range(num_layers):
            sampled_random_matrices.append(
                random_matrices[i * num_heads: (i + 1) * num_heads])
        return sampled_random_matrices
    else:
        indices = list(range(num_heads))
        # [num_layers * num_heads, proj_dim, head_dim]
        return random_matrices[indices]


def build_random_matrices(
        random_matrices: Tensor,
        tau: float,
        sigma: Optional[Tensor] = None,
        reparam_proj: bool = False) -> Tensor:
    if reparam_proj:
        random_matrices = sigma * random_matrices
    return random_matrices / tau


def _normalize(x: Tensor) -> Tuple[Tensor, Tensor]:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return torch.div(x, norm + 1e-3), norm


def random_project(
        *,
        x: Tensor,
        random_matrices: Tensor) -> Tensor:
    # x: [seq_len, bsz, num_heads, head_dim]
    # random_matrices: [num_heads, proj_dim, head_dim]

    # [1, 1, num_heads, 1]
    x, x_norm = _normalize(x)
    # [seq_len, bsz, num_heads, proj_dim]
    x = torch.einsum("sbhd,hkd->sbhk", x, random_matrices)
    x_sin, x_cos = torch.sin(x), torch.cos(x)

    # [seq_len, bsz, num_heads, 2 * proj_dim]
    phi_x = torch.cat([x_sin, x_cos], dim=-1) * SCALE
    return phi_x


def normalize_attn_weights(
        x: Tensor,
        dim: int = -1,
        dtype: torch.dtype = torch.float32) -> Tensor:
    x = x.type(torch.float32)
    # [..., 1]
    s = x.sum(dim=dim, keepdim=True).abs().clamp(1e-3)
    return torch.div(x, s).type(dtype)
