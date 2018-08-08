# -*- coding: utf-8 -*-
# @Time         : 2018-08-08 19:24
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : gradient_check.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from chapter3.mnist import load_mnist
from chapter5.two_layer_net import TwoLayerNet


def main():
    """
    对比数值方法求梯度和反向传播方法求梯度的准确性。
    由于数值方法计算速度慢，但是设计简单，编程不易出错，因此在实践中一般用来检验
    反向传播求梯度的代码有没写错
    :return:
    """
    (train_x, train_y), _ = load_mnist(one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = train_x[:3]
    y_batch = train_y[:3]

    grad_numerical = network.numerical_gradient(x_batch, y_batch)
    grad_backprop = network.gradient(x_batch, y_batch)

    for key in grad_numerical.keys():
        # 求两种梯度计算方法的差的绝对值的平均
        # 由于计算机计算时存在精度的问题，因此只要diff较小就可以认为结果一致
        diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
        print(key + ": " + str(diff))


if __name__ == '__main__':
    main()
