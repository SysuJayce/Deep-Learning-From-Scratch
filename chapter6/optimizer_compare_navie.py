# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 17:20
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : optimizer_compare_navie.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from common.optimizer import *


def f(x, y):
    """
    测试函数f=x^2 / 20 + y^2
    :param x:
    :param y:
    :return:
    """
    return x ** 2 / 20.0 + y ** 2


def df(x, y):
    """
    函数f的梯度
    :param x:
    :param y:
    :return:
    """
    return x / 10.0, 2 * y


def main():
    init_pos = (-7.0, 2.0)  # 测试起点
    params, grads = {}, {}

    # 对比这4个优化器
    optimizers = OrderedDict()
    optimizers['SGD'] = SGD(lr=0.95)
    optimizers['Momentum'] = Momentum(lr=0.1)
    optimizers['AdaGrad'] = AdaGrad(lr=1.5)
    optimizers['Adam'] = Adam(lr=0.3)

    idx = 1  # 用于matplotlib画图的时候的下标
    for key in optimizers:  # 逐个优化器进行优化并画图
        optimizer = optimizers[key]
        x_history, y_history = [], []  # 优化器的优化记录
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(30):  # 优化30步
            x_history.append(params['x'])
            y_history.append(params['y'])

            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)

        """使用matplotlib画图"""
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        mask = Z > 7
        Z[mask] = 0

        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color='red')
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key)
        plt.xlabel('x')
        plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    main()
