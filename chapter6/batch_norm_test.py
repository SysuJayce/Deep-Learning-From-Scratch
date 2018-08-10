# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 14:37
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : batch_norm_test.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from chapter3.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
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


def train(train_x, train_label, weight_init_std, learning_rate, max_epoch,
          batch_size):
    """
    构造带有BN层的神经网络和不带BN层的神经网络。测试BN层的效果
    :param train_x:
    :param train_label:
    :param weight_init_std: 参数初始化方式
    :param learning_rate:
    :param max_epoch: 测试的epoch数
    :param batch_size:
    :return:
    """
    bn_network = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=weight_init_std,
                                     use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784,
                                  hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10,
                                  weight_init_std=weight_init_std,
                                  use_batchnorm=False)

    optimizer = SGD(learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

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

        # 两个网络分别更新
        for _network in (bn_network, network):
            grads = _network.gradient(batch_x, batch_label)
            optimizer.update(_network.params, grads)

        # 每一个epoch记录一个在测试集上的准确率
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(train_x, train_label)
            bn_train_acc = bn_network.accuracy(train_x, train_label)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - "
                  + str(bn_train_acc))
            epoch_cnt += 1

    return train_acc_list, bn_train_acc_list


def draw(train_acc_list, bn_train_acc_list, i, w, x):
    """
    根据有BN层和没BN层的神经网络训练时在训练集上的准确率绘制准确率变化图
    :param train_acc_list:
    :param bn_train_acc_list:
    :param i:
    :param w:
    :param x:
    :return:
    """
    plt.subplot(4, 4, i+1)
    plt.title("W: " + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization',
                 markevery=2)
        plt.plot(x, train_acc_list, linestyle='--',
                 label='No Batch Normalization', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')

    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel('epoch')

    plt.legend(loc='lower right')


def main():
    (train_x, train_label), _ = load_mnist()
    train_x = train_x[: 1000]
    train_label = train_label[: 1000]

    max_epoch = 20
    batch_size = 100
    learning_rate = 0.01

    weight_scale_list = np.logspace(0, -4, num=16)  # 测试16种初始化方式
    x = np.arange(max_epoch)  # 画图时的x轴

    for i, w in enumerate(weight_scale_list):
        print("============== " + str(i + 1) + "/16" + " ==============")
        train_acc_list, bn_train_acc_list = train(train_x, train_label, w,
                                                  learning_rate, max_epoch,
                                                  batch_size)
        draw(train_acc_list, bn_train_acc_list, i, w, x)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
