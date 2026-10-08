from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="ADMM-NeuralNetwork",
    version="0.1.0",
    python_requires=">=3.11",
    ext_modules=cythonize(
        [Extension(f"src.cyth.{name}", [f"src/cyth/{name}.pyx"],
                   include_dirs=[np.get_include()])
         for name in ("argminc", "binarymin")],
        compiler_directives={"language_level": 3},
    ),
    packages=["src", "src.algorithms", "src.cyth"],
    install_requires=["numpy>=2.0", "scipy>=1.14", "scikit-learn>=1.5"],
    extras_require={"plot": ["matplotlib>=3.9"], "test": ["pytest>=8"]},
    entry_points={"console_scripts": ["admm-runner=src.runner:main"]},
    description="Experimental centralized ADMM approach for neural networks",
    author="Lorenzo Rutigliano",
    author_email="lnz.rutigliano@gmail.com",
)
