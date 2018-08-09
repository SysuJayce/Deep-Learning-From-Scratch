# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 0:08
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : weight_init_compare.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import matplotlib.pyplot as plt

from chapter3.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


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


def main():
    (train_x, train_label), _ = load_mnist()
    train_size = train_x.shape[0]
    batch_size = 128
    max_iterations = 2000

    # 打乱训练集顺序，每次训练随机获取一个batch大小的训练样本
    batch_mask = np.arange(train_size)
    np.random.shuffle(batch_mask)

    # 使用SGD优化器
    optimizer = SGD(lr=0.01)

    # 将要对比的权重初始化方式： 0.01， Xavier， He 三种
    weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid',
                         'He': 'relu'}

    # 为每个优化器生成一个五层全连接神经网络
    networks = {}
    train_loss_list = {}
    for key, weight_init_type in weight_init_types.items():
        networks[key] = MultiLayerNet(input_size=784,
                                      hidden_size_list=[100, 100, 100, 100],
                                      output_size=10,
                                      weight_init_std=weight_init_type)
        train_loss_list[key] = []  # 每个优化器都记录在训练集上的损失值

    left = 0
    for i in range(max_iterations):
        # 获取一个batch
        batch_x, batch_label, left = get_batch(train_x, train_label, batch_mask,
                                               batch_size, left)

        # 计算梯度，然后用不同优化器去更新参数
        # 记录每次更新后的损失值
        for key in weight_init_types.keys():
            grads = networks[key].gradient(batch_x, batch_label)
            optimizer.update(networks[key].params, grads)

            loss = networks[key].loss(batch_x, batch_label)
            train_loss_list[key].append(loss)

        # 每迭代100次就输出一次当前各优化器的损失值
        if i % 100 == 0:
            print("="*15 + "iteration: " + str(i) + "="*15)
            for key in weight_init_types.keys():
                loss = train_loss_list[key][-1]
                print(key + ": " + str(loss))

    # 绘制损失值随迭代次数变化图
    markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
    x = np.arange(max_iterations)
    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss_list[key]), marker=markers[key],
                 markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 实验效果：He > Xavier > std=0.01
    main()
