# -*- coding: utf-8 -*-
# @Time         : 2018-08-04 23:42
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : train_neural_network.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from common.mnist import load_mnist
from chapter4.two_layer_net import TwoLayerNet


def train():
    (train_x, train_label), (test_x, test_label) =\
        load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    iterations = 10000

    """
    为了实现每次随机选取一个batch_size的样本来训练，一般有2种做法
    1. 先打乱训练集的样本顺序，然后按顺序每次取batch_size个样本
    2. 不改变训练集的样本顺序，每次随机选取batch_size个样本。
    
    但是2的做法极有可能一个epoch不能遍历所有样本，因此我就想到先声明一个
    下标列表，然后打乱顺序，再按顺序访问即可
    """
    batch_mask = np.arange(train_x.shape[0])
    np.random.shuffle(batch_mask)

    batch_size = 100
    learning_rate = 0.1

    train_loss_list_ = []
    train_acc_list_ = []
    test_acc_list_ = []

    left = 0  # 按顺序取样本的时候的一个游标
    epoch_num = 0
    # 没有严格训练iteration次，而是改成了以在iteration次迭代中最大的epoch数为准
    # 这样就保证所有样本被训练的次数是一样的(一个epoch就是全部样本遍历一次)
    for i in range(int(iterations / batch_size) * batch_size):
        batch_x = train_x[batch_mask[left: left+batch_size]]
        batch_label = train_label[batch_mask[left: left+batch_size]]

        # 使用数值方法计算梯度的效率太低，跑十几个小时也才跑了1000多个iteration
        # grad = network.numerical_gradient(batch_x, batch_label)
        # 使用BP算法来计算梯度效率高太多，几十秒就能跑完10000个iteration
        grad = network.gradient(batch_x, batch_label)

        # 更新神经网络参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(batch_x, batch_label)
        train_loss_list_.append(loss)

        left += batch_size  # 在一次迭代结束时更新游标

        # 如果完成一个epoch，就计算一次准确率
        if left >= train_x.shape[0]:
            left = 0
            epoch_num += 1
            train_acc = network.accuracy(train_x, train_label)
            test_acc = network.accuracy(test_x, test_label)
            train_acc_list_.append(train_acc)
            test_acc_list_.append(test_acc)
            print("No.%d epoch:" % epoch_num)
            print("train acc: %f\ttest acc: %f" % (train_acc, test_acc))

    return train_acc_list_, test_acc_list_, train_loss_list_


def draw(train_acc_list_, test_acc_list_):
    """
    绘制训练过程中的准确率变化图
    :param train_acc_list_:
    :param test_acc_list_:
    :return:
    """
    x = np.arange(len(train_acc_list_))
    plt.plot(x, train_acc_list_, label='train acc')
    plt.plot(x, test_acc_list_, label='test acc', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    train_acc_list, test_acc_list, train_loss_list = train()
    draw(train_acc_list, test_acc_list)
