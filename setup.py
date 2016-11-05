try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize
import numpy as np


setup(
    name='ADMM-NeuralNetwork',
    ext_modules=cythonize("src/cyth/*.pyx", include_path=[np.get_include()]),
    packages=['src', 'src.algorithms', 'src.cyth'],
    install_requires=[
        'numpy >= 1.11.1',
        'scikit-learn >= 0.17.1',
        'Cython >= 0.24.1'
    ],
    entry_point={
        "console_scripts": ['admm-report = src.runner:main']
    },
    description='Experimental Centralized ADMM approach for Neural Networks',
    author='Lorenzo Rutigliano',
    author_email='lnz.rutigliano@gmail.com'
)
