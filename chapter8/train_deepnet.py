# -*- coding: utf-8 -*-
# @Time         : 2018-08-13 15:57
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : train_deepnet.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from common.mnist import load_mnist
from chapter8.deep_convnet import DeepConvNet
from common.trainer import Trainer


def draw(train_acc_list, test_acc_list):
    """
    绘制在测试集和训练集上的准确率变化图
    :param train_acc_list:
    :param test_acc_list:
    :return:
    """
    x = np.arange(len(train_acc_list))  # 画图时的x轴
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


def main():
    # 获取MNIST数据
    (train_x, train_label), (test_x, test_label) = load_mnist(flatten=False)

    # 构造深层CNN
    network = DeepConvNet()

    # 生成一个训练器
    trainer = Trainer(network, train_x, train_label, test_x, test_label,
                      epochs=20, mini_batch_size=100, optimizer='Adam',
                      optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000, verbose=True)

    trainer.train()  # 训练上面构造好的神经网络

    network.save_params()  # 训练完成后持久化参数
    print("Saved Network Parameters!")

    # 获取训练过程中训练集和测试集的准确率
    train_acc_list = trainer.train_acc_list
    test_acc_list = trainer.test_acc_list

    draw(train_acc_list, test_acc_list)  # 绘制准确率变化图


def test():
    # 获取MNIST数据
    _, (test_x, test_label) = load_mnist(flatten=False)

    network = DeepConvNet()
    network.load_params()

    test_acc = network.accuracy(test_x, test_label)
    print("test acc:", test_acc)


if __name__ == '__main__':
    main()
    # test()
