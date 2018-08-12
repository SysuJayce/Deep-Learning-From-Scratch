# -*- coding: utf-8 -*-
# @Time         : 2018-08-11 21:30
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : simple_convnet.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import pickle
import numpy as np
from collections import OrderedDict

from common.layers import Convolution, Pooling, Affine, SoftmaxWithLoss, ReLU
from common.gradient import numerical_gradient as ng


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param=None, hidden_size=100,
                 output_size=10, weight_init_std=0.01, regularizer_lambda=0.1):
        # 卷积层的默认参数：默认情况下滤波器个数为30个，大小为5x5，不填充，步长1
        if conv_param is None:
            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0,
                          'stride': 1}
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]  # 输入层的矩阵大小：单通道下二维矩阵的宽/高
        conv_output_size = int((input_size + 2 * filter_pad - filter_size) /
                               filter_stride + 1)  # 卷积层输出的单个特征图的大小
        # 最大池化层的输出大小：池化后保持特征图个数不变，由于使用的是2x2的最大
        # 池化层，因此宽/高都变为原来的一半。
        # 总的输出元素个数为：特征图个数 * (卷积层输出 / 2) * (卷积层输出 / 2)

        # 因为这里的简单CNN中池化层后面接全连接层，
        # 需要将池化层的节点拉平成一个一维数组
        pool_output_size = int(filter_num * (conv_output_size / 2) ** 2)

        self.regularizer_lambda = regularizer_lambda  # 正则化强度

        # 初始化神经网络各层的参数：卷积层、(池化层)、全连接层、全连接层
        # 其中池化层没有需要训练的参数，因此不需要初始化。
        self.params = {}
        # 第一层(卷积层)：滤波器的参数(权重参数) + 偏置参数
        # 滤波器的参数有4个：滤波器个数、通道数、高、宽
        self.params['W1'] = weight_init_std * np.random.randn(filter_num,
                                                              input_dim[0],
                                                              filter_size,
                                                              filter_size)
        # 卷积层的偏置参数：一个滤波器需要一个偏置，因此一共filter_num个偏置
        self.params['b1'] = np.zeros(filter_num)
        # 全连接层(在这里是一个隐藏层)权重参数：
        # 输入节点数为池化层的所有节点个数，输出为隐藏层大小
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,
                                                              hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        # 全连接层(在这里是输出层)权重参数：
        # 输入节点数为隐藏层的所有节点个数，输出为输出层大小
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,
                                                              output_size)
        self.params['b3'] = np.zeros(output_size)

        # 构造神经网络：
        # 卷积层、激活层(ReLU层)、最大池化层、
        # 仿射层(隐藏层)、激活层(ReLU层)、仿射层(输出层)
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # 最后加入一层SoftmaxWithLoss层用于计算交叉熵误差，帮助训练神经网络
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 逐层前向传播，预测输入x的输出
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, true_label):
        """
        计算损失。书上原本只计算了交叉熵误差，我在这里加上L2正则化
        :param x:
        :param true_label:
        :return:
        """
        # 计算交叉熵误差
        y = self.predict(x)
        total_loss = self.last_layer.forward(y, true_label)

        # 计算L2正则化项。不知为何加入L2正则化之后损失收敛不了，一直递增
        # 但是训练是正常进行的，准确率也很稳定
        regularizer = 0
        for idx in (1, 2, 3):
            W = self.params['W'+str(idx)]
            regularizer += 0.5 * self.regularizer_lambda * np.sum(W ** 2)

        total_loss += regularizer

        return total_loss

    def accuracy(self, x, true_label, batch_size=100):
        """
        计算输入x的预测准确率。使用batch处理加速运算
        :param x: 输入数据
        :param true_label: 真实标签
        :param batch_size: 批处理数据量
        :return: 准确率
        """
        # 如果真实标签是one-hot编码，先提取成一维数组：一行代表一个真实值
        if true_label.ndim != 1:
            true_label = np.argmax(true_label, axis=1)

        correct_cnt = 0.0
        # 书上原本代码没有处理剩余的这部分数据。
        # 在这里加上iters这个变量，用于控制循环次数
        if x.shape[0] % batch_size:
            iters = int(x.shape[0] / batch_size) + 1
        else:
            iters = int(x.shape[0] / batch_size)

        for i in range(iters):
            # 获取一个batch的数据和对应的真实标签
            temp_x = x[i * batch_size: (i+1) * batch_size]
            temp_true_label = true_label[i * batch_size: (i+1) * batch_size]
            # 预测这个batch的数据的输出
            y = self.predict(temp_x)
            y = np.argmax(y, axis=1)
            # 统计每个batch的预测正确数
            correct_cnt += np.sum(y == temp_true_label)

        acc = correct_cnt / x.shape[0]  # 计算准确率
        return acc

    def numerical_gradient(self, x, true_label):
        # 数值方法计算梯度
        loss_func = lambda _: self.loss(x, true_label)

        grads = {}
        for idx in range(1, 4):
            grads['W'+str(idx)] = ng(loss_func, self.params['W'+str(idx)])
            grads['b'+str(idx)] = ng(loss_func, self.params['b'+str(idx)])

        return grads

    def gradient(self, x, true_label):
        """
        反向传播计算梯度
        :param x:
        :param true_label:
        :return:
        """
        # 先前向传播计算中间值
        self.loss(x, true_label)

        """反向传播"""
        dout = 1
        dout = self.last_layer.backward(dout)  # 反向传播经过输出层的激活函数

        # 逐层反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 反向传播结束后从各层提取梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self):
        # 持久化训练好的参数
        file_path = './data/params.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self):
        # 加载参数
        file_path = './data/params.pkl'

        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            for key, val in params.items():
                # 直接将params赋值给self.params的话，
                # 改变params也会改变self.params，不安全
                self.params[key] = val
