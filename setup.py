from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "pwkrr._vec_trick",
        ["pwkrr/_vec_trick.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "pwkrr._generalized_jaccard",
        ["pwkrr/_generalized_jaccard.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="pwkrr",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
)
