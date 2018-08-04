# -*- coding: utf-8 -*-
# @Time         : 2018-08-03 16:47
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient_2d.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt


def _numerical_gradient_no_batch(f, x):
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


def numerical_gradient(f, X):
    """
    计算函数f在一个batch的X下的梯度
    :param f:
    :param X:
    :return:
    """
    if X.ndim == 1:  # 如果batch_size为1则退化成无batch情况
        return _numerical_gradient_no_batch(f, X)

    grad = np.zeros_like(X)

    for idx, x in enumerate(X):  # 对batch中的每个x分别求梯度
        grad[idx] = _numerical_gradient_no_batch(f, x)

    return grad


def func(x):
    """
    函数表达式：h = x^2 + y^2 + z^2
    :param x:
    :return:
    """
    if x.ndim == 1:  # 如果输入x为一维数组，则简单求每个分量的平方和
        return np.sum((x**2))

    return np.sum(x**2, axis=1)  # 如果输入为二维矩阵，则需要在行方向求平方和


def main():
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)  # 生成两个方阵，获得x0和x1的所有组合，用于画图

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(func, np.array([X, Y]))

    # 绘制梯度图
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    main()
