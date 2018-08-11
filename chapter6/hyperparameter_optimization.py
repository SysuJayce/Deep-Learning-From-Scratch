# -*- coding: utf-8 -*-
# @Time         : 2018-08-11 0:05
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : hyperparameter_optimization.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from chapter3.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.util import shuffle_dataset


def train(train_x, train_label, val_x, val_label, lr, weight_decay, epochs=50):
    # 按照给定的超参数进行训练一个神经网络，并返回在验证集和训练集上的准确率
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, train_x, train_label, val_x, val_label,
                      epochs=epochs, mini_batch_size=100, optimizer='SGD',
                      optimizer_param={'lr': lr}, verbose=True)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


def draw(val_result, train_result):
    # 绘制验证集和训练集上的准确率变化图
    graph_draw_num = 20  # 绘制20张子图
    # 5列4行
    col_num = 5
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0

    # 先对验证集上的准确率进行排序：由于val_result每一个key对应的value中包含
    # 50个epoch的验证集准确率以及一个final的验证集准确率，
    # 在这里我们用final的准确率来降序排序
    for key, val_acc_list in sorted(val_result.items(), key=lambda k: k[1][-1],
                                    reverse=True):
        print("Best - " + str(i+1) + " (val acc: " + str(val_acc_list[-1])
              + ") | " + key)
        plt.subplot(row_num, col_num, i+1)
        plt.title("Best - " + str(i+1))
        plt.ylim(0.0, 1.0)
        if i % 5:
            plt.yticks([])
        plt.xticks()
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)
        plt.plot(x, train_result[key], '--')
        i += 1

        if i >= graph_draw_num:
            break

    plt.show()


def main():
    # 获取MNIST数据集，为了加速测试，只使用训练集前500个样本
    (train_x, train_label), _ = load_mnist()
    train_x = train_x[: 500]
    train_label = train_label[: 500]

    # 从训练集中划分一部分作为验证集
    validation_rate = 0.2
    validation_num = int(train_x.shape[0] * validation_rate)
    # 先打乱训练集再划分
    train_x, train_label = shuffle_dataset(train_x, train_label)
    val_x = train_x[: validation_num]
    val_label = train_label[: validation_num]
    train_x = train_x[validation_num:]
    train_label = train_label[validation_num:]

    # 迭代100次来寻找最优超参数
    optimization_trial = 100
    val_result = {}
    train_result = {}

    for _ in range(optimization_trial):
        # 在指定的搜索范围随机对L2正则化强度和学习率进行采样
        weight_decay = 10 ** np.random.uniform(-8, -4)
        lr = 10 ** np.random.uniform(-6, -2)

        # 利用本次迭代采样得到的两个超参数进行训练，得到验证集和训练集上的准确率
        val_acc_list, train_acc_list = train(train_x, train_label, val_x,
                                             val_label, lr, weight_decay)
        print("val acc: " + str(val_acc_list[-1]) + " | lr: " + str(lr) +
              " | weight decay: " + str(weight_decay))
        # 把本次迭代的结果保存起来，记录所用的超参数以及测试结果
        key = "lr: " + str(lr) + ", weight decay: " + str(weight_decay)
        val_result[key] = val_acc_list
        train_result[key] = train_acc_list

    print("\n========== Hyper-Parameter Optimization Result ===========")
    draw(val_result, train_result)


if __name__ == '__main__':
    main()
