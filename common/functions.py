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


def mean_squared_error(true_label, prediction):
    return 0.5 * np.sum((true_label-prediction)**2)


def cross_entropy_error(true_label, prediction):
    delta = 1e-7  # 设置一个极小值来防止计算ln(0)时溢出下界

    # 维度为1代表向量(array数组)，维度为2代表矩阵
    # 这里将向量转换成矩阵形式
    if prediction.ndim == 1:
        true_label = true_label.reshape(1, true_label.size)
        prediction = prediction.reshape(1, prediction.size)

    # 如果预测值和真实值的大小一样，说明真实值是以one-hot编码的
    # 因此将真实值转换成one-hot编码的索引
    # 这时true_label表示预测值array中用于计算熵大小的下标
    # 即 -sum(ln(pred[true_label]))
    if true_label.size == prediction.size:
        true_label = true_label.argmax(axis=1)

    batch_size = prediction.shape[0]  # 如果是以batch进行训练就求平均误差
    return -np.sum(np.log(prediction[np.arange(batch_size), true_label] +
                          delta)) / batch_size
