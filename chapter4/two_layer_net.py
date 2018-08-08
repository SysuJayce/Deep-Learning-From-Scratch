# -*- coding: utf-8 -*-
# @Time         : 2018-08-04 22:55
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : two_layer_net.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from common.functions import softmax, cross_entropy_error, sigmoid, sigmoid_grad
from common.gradient import numerical_gradient as ng


class TwoLayerNet:
    """
    两层神经网络，使用梯度下降来优化损失函数(交叉熵误差)
    """
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化神经网络的参数
        weight_init_std = 0.01
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size,
                                                              hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,
                                                              output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        输入x然后推理预测结果
        :param x:
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, true_label):
        """
        计算交叉熵误差损失函数的输出
        :param x:
        :param true_label:
        :return:
        """
        pred = self.predict(x)
        return cross_entropy_error(true_label, pred)

    def accuracy(self, x, true_label):
        """
        计算由输入的x推理的预测结果的准确率
        :param x:
        :param true_label:
        :return:
        """
        pred = self.predict(x)
        pred = np.argmax(pred, axis=1)  # 取各输出节点中值最大的下标作为预测值
        true_label = np.argmax(true_label, axis=1)  # 同上

        accuracy = np.sum(pred == true_label) / float(x.shape[0])  # 计算准确率
        return accuracy

    def numerical_gradient(self, x, true_label):
        """
        数值方法计算梯度。虽然计算缓慢，但是实现简单。
        一般用于校验反向传播时计算的梯度的正确性
        :param x:
        :param true_label:
        :return:
        """
        def loss_func(x_):
            return self.loss(x, true_label)

        grads = dict()
        grads['W1'] = ng(loss_func, self.params['W1'])
        grads['W2'] = ng(loss_func, self.params['W2'])
        grads['b1'] = ng(loss_func, self.params['b1'])
        grads['b2'] = ng(loss_func, self.params['b2'])

        return grads

    def gradient(self, x, true_label):
        """
        使用了BP算法来计算梯度
        :param x: 输入数据
        :param true_label: 真实label
        :return: 在输入为x的时候，损失函数的梯度
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # 先前向传播计算出预测值
        # 在反向传播的时候可以直接看等号右边的表达式来想怎么传播
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        """
       根据预测值反向传播，逐步求各参数的梯度
       先是由损失函数对y求导(这时反向传播到了softmax之前，因此下一步就是
       由softmax的输入对W2/b2求导)
       反向传播每传播一步，下一步要求导的时候就是以上一层的输出作为
       函数表达式，对上一层的各个参数进行求导
       """
        # 由于使用的是softmax + 交叉熵误差损失函数，因此求导结果为y-t形式(残差)
        # 由于是使用batch加速运算，所以反向传播过程中要求平均
        dy = (y - true_label) / batch_num
        # 求完dy之后继续反向，也就是求softmax的输入a2对W2的导数，即z1
        # 在链式法则相乘的时候需要注意矩阵的转置与否，目的是保证求导前后形状一致
        grads['W2'] = np.dot(z1.T, dy)
        # 对b2求导数的时候导数为1，但是由于在正向传播的时候bias加到了一个batch中
        # 每一个样本的对应节点中，因此在反向传播计算bias的梯度的时候需要将同一列
        # 的所有元素加起来作为bias的一个梯度元素
        grads['b2'] = np.sum(dy, axis=0)

        """
       书上原本的代码是这样的，在da1和dz1的表示上不妥
       因此改成下面形式。但是原来的表示在结果上也一样的
       """
        # da1 = np.dot(dy, W2.T)
        # dz1 = sigmoid_grad(a1) * da1
        # grads['W1'] = np.dot(x.T, dz1)
        # grads['b1'] = np.sum(dz1, axis=0)

        dz1 = np.dot(dy, W2.T)
        # 这里对a1求导的时候sigmoid_grad的参数是a1
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
