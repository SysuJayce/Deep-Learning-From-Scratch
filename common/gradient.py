# -*- coding: utf-8 -*-
# @Time         : 2018-08-04 22:19
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def _numerical_gradient_1d(f, x):
    """
    给定一个变量x，计算函数f在x处的梯度。这里求的是单个x的情况，不涉及batch
    :param f:
    :param x:
    :return: 函数f在x处的梯度(一个向量)
    """
    delta = 1e-4  # 微元大小
    # 由于梯度需要对每个分量求偏导，最后得到的梯度(向量)大小和x一样
    grad = np.zeros_like(x)

    for idx in range(x.size):
        temp = x[idx]  # 保存第idx个分量的值

        # 对第idx个分量求偏导，则其他分量的值保持不变
        x[idx] = temp + delta
        fh1 = f(x)
        x[idx] = temp - delta
        fh2 = f(x)

        grad[idx] = (fh1-fh2) / (2*delta)

        x[idx] = temp  # 计算完这个分量的偏导之后还原x

    return grad


def numerical_gradient_2d(f, X):
    """
    计算函数f在一个batch的X下的梯度
    :param f:
    :param X:
    :return:
    """
    if X.ndim == 1:  # 如果batch_size为1则退化成无batch情况
        return _numerical_gradient_1d(f, X)

    grad = np.zeros_like(X)

    for idx, x in enumerate(X):  # 对batch中的每个x分别求梯度
        grad[idx] = _numerical_gradient_1d(f, x)

    return grad


def numerical_gradient(f, x):
    """
    计算多维数组的梯度(可以处理大于2维的数组)
    :param f:
    :param x:
    :return:
    """
    delta = 1e-4
    grad = np.zeros_like(x)

    # 数组x的迭代器，追踪多重索引
    # it[0]表示迭代器it所指的值
    # multi_index为多重索引，it.multi_index可以返回迭代器it当前所处的索引
    # readwrite表示允许对数组x进行读写
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        temp = x[idx]

        x[idx] = temp + delta
        fh1 = f(x)
        x[idx] = temp - delta
        fh2 = f(x)

        grad[idx] = (fh1-fh2) / (2*delta)

        x[idx] = temp
        it.iternext()

    return grad
