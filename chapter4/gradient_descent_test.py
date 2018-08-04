# -*- coding: utf-8 -*-
# @Time         : 2018-08-03 16:48
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient_descent_test.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pylab as plt

from chapter4.gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr, step_num):
    """
    梯度下降法求极小值
    :param f: 损失函数f
    :param init_x: 初始值x
    :param lr: 学习率
    :param step_num: 迭代次数
    :return:
    """
    x = init_x
    x_history = []  # 记录每一步的优化结果

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)  # 求梯度
        x -= lr * grad  # 沿梯度方向以学习率为步长进行优化

    return x, np.array(x_history)


def func(x):
    """
    本次测试的损失函数
    :param x:
    :return:
    """
    return x[0] ** 2 + x[1] ** 2


def main():
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num = 20

    x, x_history = gradient_descent(func, init_x, lr, step_num)

    """
    绘制梯度下降学习过程
    """
    plt.plot([-5, 5], [0, 0], '--b')  # 画四个象限
    plt.plot([0, 0], [-5, 5], '--b')
    
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')  # 描绘学习过程

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()


if __name__ == "__main__":
    main()
