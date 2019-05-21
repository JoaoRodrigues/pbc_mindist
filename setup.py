#!/usr/bin/env python

# Copyright 2019 João Pedro Rodrigues
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Calculation of minimum distance between periodic images in molecular
dynamics simulations.
"""

import os.path
from setuptools import setup
from setuptools.extension import Extension
import sys

try:
    import numpy as np
except ImportError as err:
    print(f'Could not import numpy: {err}', file=sys.stderr)

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False

packages = ['pbc_mindist']

ext = '.pyx' if CYTHON else '.c'
ext_modules = [Extension("pbc_mindist.dist",
                         [os.path.join("cython", "dist" + ext)],
                         include_dirs=[np.get_include()],
                         extra_compile_args=['-ffast-math', '-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         ),
               ]

cmdclass = {}
if CYTHON:
    ext_modules = cythonize(ext_modules)
    cmdclass['build_ext'] = build_ext

requirements = []

setup(name="pbc_mindist",
      version='1.0.0',
      description=__doc__,
      url="https://github.com/joaorodrigues/pbc_mindist",
      author='Joao Rodrigues',
      author_email='j.p.g.l.m.rodrigues@gmail.com',
      packages=packages,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      install_requires=requirements,
      entry_points={
          'console_scripts': [
              'pbc_mindist = pbc_mindist.pbc_mindist:main',
              ]
           },
      )
