import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'rfa_cuda', [
            'cuda/rfa_cuda.cpp',
            'cuda/forward.cu',
            'cuda/backward.cu',
            'cuda/utils.cu'
        ],
        extra_compile_args={
            'cxx': [
                '-g',
                '-v'],
            'nvcc': [
                '-DCUDA_HAS_FP16=1',
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
                '-gencode',
                'arch=compute_75,code=sm_75',
                '-use_fast_math'
            ]
        }
    )
    ext_modules.append(extension)
else:
    assert False

setup(
    name='rfa',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
