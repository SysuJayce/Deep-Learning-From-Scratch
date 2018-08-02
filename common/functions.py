# -*- coding: utf-8 -*-
# @Time         : 2018-08-02 20:17
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : functions.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def identify_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 为防止指数计算时溢出，减去最大值
    return np.exp(x) / np.sum(np.exp(x))
