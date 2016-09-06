from distutils.core import setup
from Cython.Build import cythonize
import numpy as np


setup(
    name = 'ADMM-NeuralNetwork',
    ext_modules = cythonize("src/cyth/*.pyx", include_path=[np.get_include()]),
)
