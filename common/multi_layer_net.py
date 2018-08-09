# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 17:41
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : multi_layer_net.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from collections import OrderedDict

from common.layers import *
from common.gradient import numerical_gradient as ng


class MultiLayerNet:
    """
    全连接的多层神经网络
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0):
        """
        初始化神经网络参数、构造神经网络
        :param input_size: 输入层大小
        :param hidden_size_list: 隐藏层的大小(列表每个元素代表一个隐藏层的大小)
        :param output_size: 输出层大小
        :param activation: 使用的激活函数，默认为ReLU
        :param weight_init_std: 使用的参数初始化方法
                                relu或he：使用He的初始化方法
                                sigmoid或xavier：使用Xavier的初始化方法
        :param weight_decay_lambda: L2正则化的强度参数lambda
        """
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化神经网络权重参数，初始化之后才能利用“层”构造神经网络
        self.__init_weight(weight_init_std)

        """生成神经网络"""
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}  # 可选激活函数
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            # 每一层由矩阵运算(仿射层) + 激活层 组成
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                    self.params['b'+str(idx)])
            self.layers['Activation_function'+str(idx)] =\
                activation_layer[activation]()

        # 输出层由 仿射层 + SoftmaxWithLoss层 组成
        # 因为使用的激活函数不一样，所以需要另外加入神经网络
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                self.params['b'+str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        初始化神经网络参数
        :param weight_init_std: 初始化方法
        :return:
        """
        # 记录每一层的大小，用于初始化参数
        all_size_list = [self.input_size] + self.hidden_size_list +\
                        [self.output_size]
        for idx in range(1, len(all_size_list)):
            # 根据weight_init_std选择对应的初始化方法：He/Xavier
            # 两种初始化方法的区别就在于高斯分布的标准差不同
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            else:
                raise ValueError("weight_init_std wrong.")

            # 由上述得到的标准差进行初始化
            self.params['W'+str(idx)] = scale * np.random.randn(
                all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        """
        预测（正向传播）,传播到输出层，未经过Softmax层的结果
        :param x:
        :return:
        """
        for layer in self.layers.values():  # 逐层传播即可
            x = layer.forward(x)

        return x

    def loss(self, x, true_label):
        """
        损失函数中加入L2正则化项
        :param x:
        :param true_label:
        :return:
        """
        weight_decay = 0  # L2正则化项
        # 给定的是x，需要先前向传播，计算出输出层的输出，
        # 然后由SoftmaxWithLoss计算损失
        y = self.predict(x)
        total_loss = self.last_layer.forward(y, true_label)

        # 计算所有层的L2范数：每一层的权重中的元素平方和，然后乘一个系数
        # 系数就是L2正则化的强度：0.5 * lambda
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 最终损失由SoftmaxWithLoss计算的损失 + L2正则化项
        total_loss += weight_decay
        return total_loss

    def accuracy(self, x, true_label):
        """
        计算准确率
        :param x: 输入的特征x
        :param true_label: 真实标签
        :return: 准确率
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 每个样本就一个预测值，而非一个向量

        # 如果真实标签用的是one-hot编码，那就拉成一个列向量，
        # 每一行代表一个样本的真实标签
        if true_label.ndim != 1:
            true_label = np.argmax(true_label, axis=1)

        acc = np.sum(y == true_label) / float(x.shape[0])  # 计算准确率
        return acc

    def numerical_gradient(self, x, true_label):
        """
        数值方法计算梯度
        :param x:
        :param true_label:
        :return:
        """
        loss_func = lambda _: self.loss(x, true_label)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W'+str(idx)] = ng(loss_func, self.params['W'+str(idx)])
            grads['b' + str(idx)] = ng(loss_func, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, true_label):
        """
        反向传播计算梯度
        :param x:
        :param true_label:
        :return:
        """
        # 前向传播
        self.loss(x, true_label)

        """反向传播"""
        dout = 1
        dout = self.last_layer.backward(dout)  # last_layer层的反向传播

        # 从输出层到第一个隐藏层，逐层反向传播
        # 反向传播过程自动记录了梯度，所以不用管反向传播的返回值
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 更新梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 权重参数的更新需要包括L2正则项的梯度。因为总损失是包括了L2正则项的
            grads['W'+str(idx)] = (self.layers['Affine'+str(idx)].dW +
                                   self.weight_decay_lambda *
                                   self.layers['Affine'+str(idx)].W)
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db

        return grads
