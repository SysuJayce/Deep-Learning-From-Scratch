# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 17:39
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : util.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def smooth_curve(x):
    """
    用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    使用numpy.random中的permutation函数来实现数据集的打乱
    效果和我之前自己实现的get_data是一样的，不过这个函数是直接打乱数据集，而我是
    不打乱数据集，而是打乱下标。感觉我的做法效率应该更高，不过实现会比较麻烦
    :param x: 特征
    :param t: 标签
    :return:
    """
    permutation = np.random.permutation(x.shape[0])  # 打乱下标
    # 按打乱后的下标调整数据集
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]
