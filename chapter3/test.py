# -*- coding: utf-8 -*-
# @Time         : 2018-08-01 16:49
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : test.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    # 由于numpy的函数有广播功能
    # 与标量计算的时候可以将标量广播成对应大小的数组然后运算
    y = 1.0 / (1 + np.exp(-x))
    return y


def relu(x):
    return np.maximum(x, 0)


def plot_fig(x, y):
    plt.plot(x, y)
    plt.show()


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    plot_fig(x, y1)
    plot_fig(x, y2)
    plot_fig(x, y3)


if __name__ == '__main__':
    main()
