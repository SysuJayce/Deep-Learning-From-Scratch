# -*- coding: utf-8 -*-
# @Time         : 2018-08-03 16:20
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient_1d.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    """
    使用数值微分方法求函数f在x处的导数
    :param f:
    :param x:
    :return:
    """
    delta = 1e-4
    return (f(x+delta) - f(x-delta)) / (2 * delta)


def func(x):
    return 0.01 * x**2 + 0.1 * x


def tangent_line(f, x):
    """
    求函数f在x点的切线表达式
    :param f: 原函数
    :param x: 待求切线的x值
    :return: lambda表达式，即切线表达式
    """
    k = numerical_diff(f, x)  # 切线斜率(f在x处的导数)
    print("differential of f:", k)
    b = f(x) - k * x  # 切线截距
    return lambda t: k * t + b


def main():
    x = np.arange(0.0, 20.0, 0.1)
    y = func(x)

    tf = tangent_line(func, 5)  # func在x=5的切线表达式
    y2 = tf(x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


if __name__ == '__main__':
    main()
