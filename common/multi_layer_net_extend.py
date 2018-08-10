# -*- coding: utf-8 -*-
# @Time         : 2018-08-10 12:36
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : multi_layer_net_extend.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from collections import OrderedDict

from common.layers import *
from common.gradient import numerical_gradient as ng


class MultiLayerNetExtend:
    """
    扩展版的全连接的多层神经网络
    具有L2正则化、Dropout、Batch Normalization的功能
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0, use_dropout=False, dropout_ratio=0.5,
                 use_batchnorm=False):
        """

        :param input_size: 输入层大小
        :param hidden_size_list: 隐藏层大小的列表，每个元素代表一个隐藏层的大小
        :param output_size: 输出层大小
        :param activation: 激活函数类型：relu, sigmoid
        :param weight_init_std: 参数初始化方式：relu/he, sigmoid/xavier
        :param weight_decay_lambda: L2范数的强度
        :param use_dropout: 是否使用dropout层
        :param dropout_ratio: dropout的比例
        :param use_batchnorm: 是否使用BN层
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.hidden_size_list = hidden_size_list
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化参数。需要先初始化，再调用layer中的层来构造神经网络
        self.__init_weight(weight_init_std)

        # 神经网络层。是一个有序字典，可以记录添加层的顺序，有助于前向/后向传播
        self.layers = OrderedDict()
        # 激活层所使用的激活函数
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        # 初始化所有隐藏层，输出层在之后另外添加
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                    self.params['b'+str(idx)])

            # 是否使用BN层。这里的BN层插入在激活层之前
            if use_batchnorm:
                # 初始化BN层的两个参数：gamma, beta
                self.params['gamma'+str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta'+str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+str(idx)] = BatchNormalization(
                    self.params['gamma'+str(idx)], self.params['beta'+str(idx)])

            self.layers['Activation_function'+str(idx)] =\
                activation_layer[activation]()

            # 是否使用Dropout层。Dropout层插入在激活层之后
            if use_dropout:
                self.layers['Dropout'+str(idx)] = Dropout(dropout_ratio)

        # 添加输出层。由于MNIST是分类任务，测试时输出层这里不需要使用激活函数，
        # 因此与隐藏层分开添加
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)],
                                                self.params['b'+str(idx)])
        # 训练阶段用到的最后一层：可以计算损失的Softmax激活层。
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        初始化参数
        :param weight_init_std: 初始化方式
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        :return:
        """
        # 记录所有层的大小。因为使用He和Xavier初始化方式的时候要用到前一层的大小
        all_size_list = [self.input_size] + self.hidden_size_list +\
                        [self.output_size]

        for idx in range(1, len(all_size_list)):  # 对所有层的参数进行初始化
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            else:  # 默认情况下使用的高斯分布标准差为0.01
                scale = 0.01

            # 初始化每一层的W和b参数
            self.params['W'+str(idx)] = scale * np.random.randn(
                all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flag=False):
        """
        预测输入x的输出
        :param x: 输入数据
        :param train_flag: 当前是处于训练阶段还是测试阶段。默认为测试阶段
        处于哪个阶段对于BN层和Dropout层的前向传播有影响。
        BN层在测试阶段的前向传播利用已经训练好的均值方差；
        Dropout层在测试阶段的前向传播需要激活所有神经元，但是要给予一定的抑制
        :return: 预测结果
        """
        # 按顺序逐层前向传播。注意这里不包括last_layer: SoftmaxWithLoss层
        for key, layer in self.layers.items():
            # 特殊处理Dropout层和BN层
            if key.startswith(('Dropout', 'BatchNorm')):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, true_label, train_flag=False):
        """
        计算损失值
        :param x: 输入数据
        :param true_label: 真实标签
        :param train_flag: 处于训练阶段还是测试阶段。默认为测试阶段
        :return: 预测结果和真实标签的损失值：交叉熵误差 + L2正则化项
        """
        y = self.predict(x, train_flag)  # 先进行前向传播
        # 利用预测结果求交叉熵误差
        total_loss = self.last_layer.forward(y, true_label)

        """求L2正则化项"""
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        total_loss += weight_decay  # 总损失 = 交叉熵误差 + L2正则化项
        return total_loss

    def accuracy(self, x, true_label, train_flag=False):
        """
        计算准确率
        :param x: 输入数据
        :param true_label: 真实标签
        :param train_flag: 处于训练阶段还是测试阶段。默认为测试阶段
        :return: 准确率
        """
        y = self.predict(x, train_flag)  # 先进行预测
        y = np.argmax(y, axis=1)  # 然后提取预测结果。每行一个

        if true_label.ndim != 1:  # 如果真实标签使用one-hot编码，提取真实值
            true_label = np.argmax(true_label, axis=1)

        acc = np.sum(y == true_label) / float(x.shape[0])
        return acc

    def numerical_gradient(self, x, true_label):
        """
        数值方法求梯度
        :param x:
        :param true_label:
        :return:
        """
        loss_func = lambda _: self.loss(x, true_label, train_flag=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W'+str(idx)] = ng(loss_func, self.params['W'+str(idx)])
            grads['b'+str(idx)] = ng(loss_func, self.params['b'+str(idx)])

            # BN层的gamma和beta参数的梯度
            # 但是没搞懂后面这个条件，如果输出层是BN层那就不更新这个BN层的意思？
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma'+str(idx)] = ng(loss_func,
                                             self.params['gamma'+str(idx)])
                grads['beta'+str(idx)] = ng(loss_func,
                                            self.params['beta'+str(idx)])

        return grads

    def gradient(self, x, true_label):
        """
        反向传播求梯度
        :param x:
        :param true_label:
        :return:
        """
        self.loss(x, true_label, train_flag=True)  # 先前向传播

        """反向传播"""
        # last_layer的反向传播，因为不在self.layers里面，所以先单独处理
        dout = 1
        dout = self.last_layer.backward(dout)

        # 逐层传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 反向传播结束后，Affine层和BN层中已经记录了参数的梯度，直接提取即可
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 要注意W的梯度。由于损失函数是交叉熵误差 + L2正则化项
            # 交叉熵误差对应的梯度就是Affine层里面的dW
            # L2正则项对应的梯度是 lambda * W
            # 两个梯度相加才是W最后的梯度
            grads['W'+str(idx)] = (self.layers['Affine'+str(idx)].dW +
                                   self.weight_decay_lambda *
                                   self.params['W'+str(idx)])
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db

            # 后面的条件的意思应该是BN层不能作为输出层
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma'+str(idx)] =\
                    self.layers['BatchNorm'+str(idx)].dgamma
                grads['beta'+str(idx)] = self.layers['BatchNorm'+str(idx)].dbeta

        return grads
