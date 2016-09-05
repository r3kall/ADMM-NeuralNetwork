from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("src/cyth/*.pyx"),
    name='ADMM-NeuralNetwork',
    version='0.1.0',
    packages=['src', 'src.algorithms', 'tests', 'src.cyth'],
    url='',
    license='',
    author='lnz',
    author_email='lnz.rutigliano@gmail.com',
    description='',
    install_requires=[
        'numpy',
        'scipy'
    ]
)
