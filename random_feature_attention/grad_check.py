import torch
from rfa import RFA


device = torch.device("cuda:0")
dtype = torch.half


def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)


def equal(a, b, threshold=1e-4):
    return (a - b) ** 2 < threshold


def test_forward():
    bsz, head_dim, proj_dim, tgt_len = 1, 64, 128, 50
    torch.manual_seed(3)
    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype) - .5

    rfa_func = RFA()
    attn_torch = rfa_func.forward_torch(q, k, v)
    attn_cuda = rfa_func.forward_cuda(q, k, v)
    e = (equal(attn_torch, attn_cuda))
    print(torch.all(e))


def test_backward():
    torch.manual_seed(3)
    bsz, head_dim, proj_dim = 5, 64, 128
    tgt_len = 100
    q = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    k = torch.rand(tgt_len, bsz, proj_dim, device=device, dtype=dtype) - .5
    v = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype) - .5
    grad_attn = torch.rand(tgt_len, bsz, head_dim, device=device, dtype=dtype)

    q, k, v, grad_attn = normalize(q), normalize(k), normalize(v), normalize(grad_attn)
    rfa_func = RFA()

    gq_torch, gk_torch, gv_torch = rfa_func.backward_torch(q, k, v, grad_attn)
    gq_cuda, gk_cuda, gv_cuda = rfa_func.backward_cuda(q, k, v, grad_attn)
    e = (equal(gq_cuda, gq_torch))
    print(torch.all(e))
    e = (equal(gk_cuda, gk_torch))
    print(torch.all(e))
    e = (equal(gv_cuda, gv_torch))
    print(torch.all(e))


if __name__ == "__main__":
    test_forward()
    test_backward()
