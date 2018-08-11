# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 23:58
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : overfit_dropout.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from common.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer


def draw(train_acc_list, test_acc_list):
    """
    绘制在测试集和训练集上的准确率变化图
    :param train_acc_list:
    :param test_acc_list:
    :return:
    """
    x = np.arange(len(train_acc_list))  # 画图时的x轴
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

    # 设定是否使用Dropuout，以及比例 ========================
    use_dropout = False
    # use_dropout = True
    dropout_ratio = 0.2

    network = MultiLayerNetExtend(
        input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10, use_dropout=use_dropout, dropout_ratio=dropout_ratio)

    trainer = Trainer(network, train_x, train_label, test_x, test_label,
                      epochs=301, mini_batch_size=100, optimizer='sgd',
                      optimizer_param={'lr': 0.01}, verbose=True)

    trainer.train()

    train_acc_list = trainer.train_acc_list
    test_acc_list = trainer.test_acc_list

    draw(train_acc_list, test_acc_list)


if __name__ == '__main__':
    main()
