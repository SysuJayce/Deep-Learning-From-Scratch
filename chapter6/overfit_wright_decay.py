# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 15:46
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : overfit_wright_decay.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from chapter3.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


def get_batch(train_x, train_label, batch_mask, batch_size, left):
    """
    获取一个batch的训练样本，并修改left。本质上是从打乱后的数据集的样本下标中
    选择下标，然后从数据集中根据选中的下标取出样本

    需要注意的是当剩余样本不够一个batch的这种情况，
    把这一个epoch剩余的样本选中之后从下一个epoch的起点选缺少的样本

    以及当left刚好为train_size的情况，这时需要将left更新为0，否则会下标越界报错
    :param train_x: 训练集特征
    :param train_label: 训练集的标签
    :param batch_mask: 打乱后的训练集的样本的下标
    :param batch_size: 一个batch的大小
    :param left: 选样本的下标起点
    :return: 一个batch的特征、标签以及修改后的下一个batch的起点left
    """
    train_size = train_x.shape[0]
    if left + batch_size > train_size:  # 当剩余的样本数不足一个batch时
        x_1 = train_x[batch_mask[left: train_size]]  # 取出剩余的样本
        # 从新epoch中取出缺少的样本
        x_2 = train_x[batch_mask[: batch_size-x_1.shape[0]]]
        result_x = np.concatenate((x_1, x_2), axis=0)  # 在纵向拼接起来

        y_1 = train_label[batch_mask[left: train_size]]
        y_2 = train_label[batch_mask[: batch_size - y_1.shape[0]]]
        result_y = np.concatenate((y_1, y_2), axis=0)

        left = x_2.shape[0]  # 更新起点left
        return result_x, result_y, left
    batch_x = train_x[batch_mask[left: left + batch_size]]
    batch_label = train_label[batch_mask[left: left + batch_size]]
    # 在更新left时需要注意不能越界(当更新后为train_size的时候需要置为0)
    left = left + batch_size if left + batch_size < train_size else 0
    return batch_x, batch_label, left


def train(train_x, train_label, test_x, test_label, learning_rate, max_epoch,
          batch_size):
    # weight decay（权值衰减——L2正则项强度）的设定 =======================
    weight_decay_lambda = 0  # 不使用权值衰减的情况
    # weight_decay_lambda = 0.1

    # 构造神经网络
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay_lambda)

    optimizer = SGD(learning_rate)

    train_acc_list = []
    test_acc_list = []

    train_size = train_x.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    batch_mask = np.arange(train_size)
    np.random.shuffle(batch_mask)

    epoch_cnt = 0
    left = 0
    iteration = int(iter_per_epoch * max_epoch)

    for i in range(iteration):
        # 获取一个batch的数据，更新left值
        batch_x, batch_label, left = get_batch(train_x, train_label, batch_mask,
                                               batch_size, left)

        grads = network.gradient(batch_x, batch_label)
        optimizer.update(network.params, grads)

        # 每一个epoch记录一个在测试集上的准确率
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(train_x, train_label)
            test_acc = network.accuracy(test_x, test_label)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("epoch:" + str(epoch_cnt) + ", train acc:" +
                  str(train_acc) + ", test acc:" + str(test_acc))
            epoch_cnt += 1

    return train_acc_list, test_acc_list


def draw(train_acc_list, test_acc_list, x):
    """
    绘制在测试集和训练集上的准确率变化图
    :param train_acc_list:
    :param test_acc_list:
    :param x:
    :return:
    """
    markers = {'train': 'o', 'test': 's'}
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


def main():
    (train_x, train_label), (test_x, test_label) = load_mnist()
    # 为了再现过拟合，减少学习数据
    train_x = train_x[: 300]
    train_label = train_label[: 300]

    max_epoch = 201
    batch_size = 100
    learning_rate = 0.01

    x = np.arange(max_epoch)  # 画图时的x轴

    train_acc_list, bn_train_acc_list = train(train_x, train_label, test_x,
                                              test_label, learning_rate,
                                              max_epoch, batch_size)
    draw(train_acc_list, bn_train_acc_list, x)


if __name__ == '__main__':
    main()
