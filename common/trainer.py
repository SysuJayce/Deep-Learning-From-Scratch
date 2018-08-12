# -*- coding: utf-8 -*-
# @Time         : 2018-08-11 0:12
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     batch_maskrainer.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from common.util import shuffle_dataset
from common.optimizer import *


class Trainer:
    """
    用于进行神经网络的训练的类
    """
    def __init__(self, network, train_x, train_label, test_x, test_label,
                 epochs=24, mini_batch_size=100, optimizer='SGD',
                 optimizer_param=None,
                 evaluate_sample_num_per_epoch=None, verbose=True):
        # 传给优化器的参数字典，默认情况下只传入一个学习率(默认值为0.01)
        if optimizer_param is None:
            optimizer_param = {'lr': 0.01}

        # 初始化训练类的值
        self.network = network  # 使用的神经网络
        self.verbose = verbose  # 是否打印中间过程的输出
        self.train_x = train_x  # 训练集特征
        self.train_label = train_label  # 训练集标签
        self.test_x = test_x  # 测试集特征
        self.test_label = test_label  # 测试集标签
        self.epochs = epochs  # 训练多少个epoch
        self.batch_size = mini_batch_size  # 一个batch的大小
        # 每个epoch需要用多少个样本来评估训练集和测试集的准确率
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 可用的优化器
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adam': Adam,
                                'adagrad': AdaGrad, 'rmsprpo': RMSProp}
        # 初始化优化器。根据用户输入来选择对应的优化器，
        # 然后将所需参数以字典形式传入
        self.optimizer = optimizer_class_dict[optimizer.lower()](
            **optimizer_param)

        self.train_size = train_x.shape[0]  # 训练集大小
        # 每个epoch需要多少次迭代
        self.iter_per_epoch = max(1, self.train_size / self.batch_size)
        # 最大迭代次数 = 每个epoch所需迭代次数 * epoch数
        self.max_iter = int(self.iter_per_epoch * self.epochs)
        self.current_iter = 0  # 记录当前迭代次数
        self.current_epoch = 0  # 记录当前epoch数
        self.current_index = 0  # 记录取一个batch的data时的起点下标

        # 训练时的误差、准确率记录
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def get_batch_data(self):
        """
        从训练集中依次取一个batch的数据
        :return:
        """
        next_index = self.current_index + self.batch_size  # 下标终点
        if next_index <= self.train_size:  # 如果这个batch的数据都在同一个epoch
            batch_x = self.train_x[self.current_index: next_index]
            batch_label = self.train_label[self.current_index: next_index]
            # 更新下标，注意要取模，防止当下标终点位于一个epoch的终点时下一次
            # 取数据会出现下标越界错误
            self.current_index = next_index % self.train_size
        else:  # 如果这个batch的数据需要用到旧epoch和新epoch的数据
            x1 = self.train_x[self.current_index:]  # 把旧epoch的数据取完
            next_index = self.batch_size - x1.shape[0]  # 新epoch的下标终点
            x2 = self.train_x[: next_index]  # 从新epoch中取出缺少的数据
            y1 = self.train_label[self.current_index:]
            y2 = self.train_label[: next_index]
            # 纵向合并两次取出的数据。
            batch_x = np.concatenate((x1, x2), axis=0)
            batch_label = np.concatenate((y1, y2), axis=0)
            self.current_index = next_index  # 更新下标

        return batch_x, batch_label

    def train_step(self):
        """
        单步训练函数
        :return:
        """
        batch_x, batch_label = self.get_batch_data()  # 取出一个batch的数据
        grads = self.network.gradient(batch_x, batch_label)  # 计算梯度
        # 然后把梯度传给优化器，优化器对神经网络参数进行更新
        self.optimizer.update(self.network.params, grads)

        # 训练完一个epoch后从数据集和测试集中选取evaluate_sample_num_per_epoch个
        # 样本进行评估：计算采样集的准确率(可能是为了及时监控过拟合情况)
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # 如果没有指定evaluate_sample_num_per_epoch，
            # 默认就是将全部数据用于评估
            train_x_sample, train_label_sample = self.train_x, self.train_label
            test_x_sample, test_label_sample = self.test_x, self.test_label
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                # 书上原本是每次都取前t个，我在这里修改成随机取t个
                train_batch_mask = np.random.choice(self.train_size, t)
                train_x_sample = self.train_x[train_batch_mask]
                train_label_sample = self.train_label[train_batch_mask]
                test_batch_mask = np.random.choice(self.test_x.shape[0], t)
                test_x_sample = self.test_x[test_batch_mask]
                test_label_sample = self.test_label[test_batch_mask]

            # 计算评估结果
            train_acc = self.network.accuracy(train_x_sample,
                                              train_label_sample)
            test_acc = self.network.accuracy(test_x_sample,
                                             test_label_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("====== epoch: " + str(self.current_epoch) +
                      "   |  train acc: " + str(train_acc) +
                      "   |  test acc: " + str(test_acc) + " ======")

        # 计算训练完这一步之后的损失值
        loss = self.network.loss(batch_x, batch_label)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss: " + str(loss))

        self.current_iter += 1

    def train(self):
        # 书上没有利用已经有的这个shuffle函数，我觉得可以用上。用于打乱训练集
        self.train_x, self.train_label = shuffle_dataset(self.train_x,
                                                         self.train_label)
        for i in range(self.max_iter):  # 训练max_iter次
            self.train_step()

        # 训练结束后计算测试集的准确率
        test_acc = self.network.accuracy(self.test_x, self.test_label)

        # 如果是verbose模式，输出测试结果
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc: " + str(test_acc))
