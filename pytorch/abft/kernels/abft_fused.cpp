#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> abft_fused_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    c10::optional<torch::Tensor> bias);

std::vector<torch::Tensor> abft_fused(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
  TORCH_CHECK(c.is_cuda(), "c must be CUDA tensor");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a/b/c must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "a/b shape mismatch");
  TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1), "c shape mismatch");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias.value().size(0) == c.size(1), "bias size mismatch");
  }
  return abft_fused_cuda(a, b, c, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abft_fused", &abft_fused, "Fused ABFT corner/sumc (CUDA)");
}
