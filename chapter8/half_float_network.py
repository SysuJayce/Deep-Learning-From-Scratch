# -*- coding: utf-8 -*-
# @Time         : 2018-08-13 17:11
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : half_float_network.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from chapter8.deep_convnet import DeepConvNet
from common.mnist import load_mnist


def main():
    _, (test_x, test_label) = load_mnist(flatten=False)

    sampled = 1000  # 为了实现高速化，减小样本规模
    test_x = test_x[:sampled]
    test_label = test_label[:sampled]

    network = DeepConvNet()  # 构造神经网络
    network.load_params()  # 加载已经训练好的参数

    print("caluculate accuracy (float64) ... ")
    print(network.accuracy(test_x, test_label))

    # 转换为float16型
    test_x = test_x.astype(np.float16)
    for key, val in network.params.items():
        network.params[key] = val.astype(np.float16)

    print("caluculate accuracy (float16) ... ")
    print(network.accuracy(test_x, test_label))


if __name__ == '__main__':
    main()
