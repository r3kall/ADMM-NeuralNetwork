# ADMM-NeuralNetwork
Experimental centralized ADMM approach to neural networks.

# Requirements
Numpy >= 1.11
Scipy >= 0.18
Scikit-learn >= 0.18
Cython >= 0.24

# Use without installation

First build the cython files:
```
python3.5 setup.py build_ext --inplace
```
Then you can run the program with:
```
python3.5 admm-runner.py [iris | digits]
```


# Use with installation
```
python3.5 setup.py install
```
After setup:
```
admm-report [iris | digits]
```
