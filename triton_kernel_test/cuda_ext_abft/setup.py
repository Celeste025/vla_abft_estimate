from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


this_dir = Path(__file__).resolve().parent

setup(
    name="abft_handwritten_cuda",
    ext_modules=[
        CUDAExtension(
            name="abft_handwritten_cuda",
            sources=[
                str(this_dir / "abft_cuda" / "bindings.cpp"),
                str(this_dir / "abft_cuda" / "gemm_tc_sm89.cu"),
                str(this_dir / "abft_cuda" / "gemm_tc_sm89_abft.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

