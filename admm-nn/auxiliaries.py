#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

import numpy as np


def relu(x):
    return np.maximum(0, x)
