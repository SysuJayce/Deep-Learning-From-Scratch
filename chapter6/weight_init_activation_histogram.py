# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 23:21
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : weight_init_activation_histogram.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def main():
    input_data = np.random.randn(1000, 100)  # 生成1000个样本，每个样本100维特征
    node_num = 100  # 每层神经网络的神经元个数
    hidden_layer_size = 5  # 5层神经网络

    # 使用4种初始化方法来测试
    stds = [1, 0.01, np.sqrt(1.0 / node_num), np.sqrt(2.0 / node_num)]

    for std in stds:
        plt.figure()
        activations = {}
        x = input_data
        for i in range(hidden_layer_size):
            if i != 0:  # 第一层使用生成的数据，后面都用前一层的输出作为输入
                x = activations[i - 1]

            w = std * np.random.randn(node_num, node_num)
            a = np.dot(x, w)
            # z = sigmoid(a)
            # z = ReLU(a)
            z = tanh(a)
            activations[i] = z

        # 绘制激活函数输出的分布直方图
        for i, a in activations.items():
            plt.subplot(1, len(activations), i + 1)
            plt.title(str(i + 1) + "-layer")
            if i != 0:
                plt.yticks([], [])
            plt.hist(a.flatten(), 30, range=(0, 1))

    plt.show()


if __name__ == '__main__':
    main()
