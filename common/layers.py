# -*- coding: utf-8 -*-
# @Time         : 2018-08-05 21:05
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : layers.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from common.functions import *


class ReLU:
    """
    默认输入和输出都是(1维或多维)数组
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # True / False。给数组x中非正的位置做标记
        out = x.copy()
        out[self.mask] = 0  # 把mask标记的元素置零，未标记的不变

        return out

    def backward(self, dout):
        """
        后向传播可以理解为一个函数，输入为dout，然后和relu层的偏导相乘，输出结果
        偏导只有0和1两种情况，所以输出要么保持不变，要么变为0
        :param dout:
        :return:
        """
        # 利用前向传播时做的标记给标记了的位置置零，未标记的部分置1
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)

        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None  # 记录张量的形状，用于反向传播时reshape

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # 张量的第一个维度是样本个数
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原张量形状

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 真实label

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.t, self.y)

        return self.loss

    def backward(self, dout=1):
        """
        softmax 和 交叉熵结合后的求导结果就是残差：y-t
        :param dout:
        :return:
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 当真实值是one-hot时直接对应元素相减
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            # 不是one-hot时将dx每一行中的对应真实值的下标的元素减一
            # 因为dx(即softmax的输出y)每一行的元素个数是输出层节点个数
            # 如果是one-hot那就对应下标的元素相减，如果不是one-hot
            # 那就找到真实值对应的dx的下标的元素值，然后减一
            # (减一是因为one-hot也是真实值对应的下标的元素减一)
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size  # 最后要注意batch训练的话，反向传播需要求平均

        return dx
