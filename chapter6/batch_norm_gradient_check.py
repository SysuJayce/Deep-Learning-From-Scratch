# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 22:58
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : batch_norm_gradient_check.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from common.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend


def main():
    # 读入数据
    (train_x, train_label), _ = load_mnist(one_hot_label=True)

    # 构造神经网络
    network = MultiLayerNetExtend(input_size=784,
                                  hidden_size_list=[100, 100],
                                  output_size=10,
                                  use_batchnorm=True)

    # 仅用一个训练样本来测试
    batch_x = train_x[: 1]
    batch_label = train_label[: 1]

    # 用反向传播和数值方法分别计算梯度
    grad_backprop = network.gradient(batch_x, batch_label)
    grad_numerical = network.numerical_gradient(batch_x, batch_label)

    # 比较两种方法的计算结果
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))


if __name__ == '__main__':
    main()
