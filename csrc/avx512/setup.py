from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

extra_compile_args = [
    "-O3",
    "-march=native",
    "-mavx512f",
    "-mavx512dq",
    "-mavx512bw",
    "-mavx512vl",
    "-mavx512vpopcntdq",
    "-fopenmp",
]

setup(
    name="weft_avx512",
    ext_modules=[
        CppExtension(
            name="weft_avx512",
            sources=["csrc/avx512/subbit_avx512_gemv.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
