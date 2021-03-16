#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
typedef torch::Tensor Tensor;


Tensor RFAForward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v);


std::vector<Tensor> RFABackward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& grad_attn);


Tensor forward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v) {
    return RFAForward(q, k, v);
}


std::vector<Tensor> backward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& grad_attn) {
    return RFABackward(q, k, v, grad_attn);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RFA Forward");
    m.def("backward", &backward, "RFA Backward");
}
