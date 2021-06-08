from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='quant',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'quant.cpp_extension.calc_quant_bin',
              ['quant/cpp_extension/calc_quant_bin.cc']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
