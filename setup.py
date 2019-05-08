from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [
    Extension(
        "pbc_dist",
        ["pbc_dist.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    # ext_modules=cythonize(ext_modules, annotate=True)
    ext_modules=cythonize(ext_modules)
)
