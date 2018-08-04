# -*- coding: utf-8 -*-
# @Time         : 2018-08-04 22:39
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient_simple_net.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    """
    具有预测和计算损失功能的简单网络
    """
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 从正态分布随机采样来预定义网络的参数

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, true_lable):
        """
        输入x和真实label，返回预测值与真实值之间的交叉熵误差
        :param x: 输入x
        :param true_lable: 真实label
        :return: 误差值
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(true_label=true_lable, prediction=y)

        return loss


def main():
    x = np.array([0.6, 0.9])
    label = np.array([0, 0, 1])
    net = SimpleNet()

    def func(x_):
        return net.loss(x_, label)

    dw = numerical_gradient(func, x)
    print(dw)


if __name__ == '__main__':
    main()
