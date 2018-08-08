# -*- coding: utf-8 -*-
# @Time         : 2018-08-08 16:09
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : two_layer_net.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from collections import OrderedDict

from common.layers import *
from common.gradient import numerical_gradient as ng


# noinspection PyDictCreation
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,
                                                              hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,
                                                              output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成神经网络
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 将softmax及输出层与前面的神经网络层分隔开
        # 因为在推理的时候并不需要用到这层
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        数值方法求梯度
        :param x:
        :param t:
        :return:
        """
        loss_func = lambda _: self.loss(x, t)

        grads = {}
        grads['W1'] = ng(loss_func, self.params['W1'])
        grads['b1'] = ng(loss_func, self.params['b1'])
        grads['W2'] = ng(loss_func, self.params['W2'])
        grads['b2'] = ng(loss_func, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """
        反向传播求梯度
        :param x:
        :param t:
        :return:
        """
        # 先进行前向传播，求出损失值
        # 由于最后一层是softmax+交叉熵层，在该层中已经保存了损失值
        # 所以不用设置变量接收返回值
        self.loss(x, t)

        """反向传播求梯度"""
        dout = 1  # 反向传播一般先输入1，然后逐层反向传播
        dout = self.last_layer.backward(dout)  # 倒数第一层的反向传播结果

        # 获取神经网络的所有层，然后反转，接着逐层求导反向传播
        # 由于Affine层保存了参数的导数，在反向传播过程中就可以记录梯度，
        # 所以不用管dout，直接传入下一层即可
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 从Affine层获取参数梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
