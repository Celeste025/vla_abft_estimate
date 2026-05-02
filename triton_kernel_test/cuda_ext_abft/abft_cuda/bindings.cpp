#include <torch/extension.h>

torch::Tensor gemm_tc_sm89(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_tc_sm89_abft(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sum_a_partial,
    torch::Tensor sum_b_partial,
    torch::Tensor sum_c_partial);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_tc_sm89", &gemm_tc_sm89, "Handwritten GEMM (FP16 TC, sm89)");
  m.def("gemm_tc_sm89_abft", &gemm_tc_sm89_abft, "Handwritten GEMM+ABFT (partial buffers)");
}

