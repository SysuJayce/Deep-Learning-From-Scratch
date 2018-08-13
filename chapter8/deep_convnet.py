# -*- coding: utf-8 -*-
# @Time         : 2018-08-13 13:11
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : deep_convnet.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import pickle
from collections import OrderedDict

from common.layers import Convolution, ReLU, Pooling, Dropout, Affine, \
                            SoftmaxWithLoss


class DeepConvNet:
    """
    识别率为99%以上的高精度的ConvNet
        网络结构如下所示
            conv - relu - conv- relu - pool -
            conv - relu - conv- relu - pool -
            conv - relu - conv- relu - pool -
            affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28), conv_param_1=None,
                 conv_param_2=None, conv_param_3=None, conv_param_4=None,
                 conv_param_5=None, conv_param_6=None, hidden_size=50,
                 output_size=10):
        # 第一个卷积层输入1x28x28，输出16x28x28
        if conv_param_1 is None:
            conv_param_1 = {'filter_num': 16, 'filter_size': 3, 'pad': 1,
                            'stride': 1}
        # 第二个卷积层输入16x28x28，输出16x28x28
        if conv_param_2 is None:
            conv_param_2 = {'filter_num': 16, 'filter_size': 3, 'pad': 1,
                            'stride': 1}
        # 第二个卷积层之后接最大池化层，池化层大小为2x2,步长为2，即高、宽减半
        # 第三个卷积层输入16x14x14，输出32x14x14
        if conv_param_3 is None:
            conv_param_3 = {'filter_num': 32, 'filter_size': 3, 'pad': 1,
                            'stride': 1}
        # 第四个卷积层输入32x14x14，但由于pad2个，因此输出32x16x16
        if conv_param_4 is None:
            conv_param_4 = {'filter_num': 32, 'filter_size': 3, 'pad': 2,
                            'stride': 1}
        # 第四个卷积层之后接最大池化层，池化层大小为2x2,步长为2，即高、宽减半
        # 第五个卷积层输入32x8x8，输出64x8x8
        if conv_param_5 is None:
            conv_param_5 = {'filter_num': 64, 'filter_size': 3, 'pad': 1,
                            'stride': 1}
        # 第五个卷积层输入64x8x8，输出64x8x8
        if conv_param_6 is None:
            conv_param_6 = {'filter_num': 64, 'filter_size': 3, 'pad': 1,
                            'stride': 1}

        """
        卷积层的每个节点只与前一层的filter_size个节点连接，
        即本层卷积层的卷积核 高x宽有多少，就和前一层的多少个节点连接。
        如果有多个通道，那还要乘上通道数(深度)
        这里的所有卷积层都用3x3的大小
        
        各层输出如下：
        卷积层1：              16 28 28
        卷积层2 | 池化层1：    16 28 28 | 16 14 14
        卷积层3：              32 14 14
        卷积层4 | 池化层2：    32 16 16 | 32 8 8
        卷积层5：              64 8 8
        卷积层6：              64 8 8 | 64 4 4
        """
        pre_node_nums = np.array(
            [1 * 3 * 3,  # 卷积层1：前一层(输入层)通道数(深度)为1
             16 * 3 * 3,  # 卷积层2：前一层(卷积层1)通道数(深度)为16
             16 * 3 * 3,  # 卷积层3：前一层(卷积层2)通道数(深度)为16
             32 * 3 * 3,  # 卷积层4：前一层(卷积层3)通道数(深度)为32
             32 * 3 * 3,  # 卷积层5：前一层(卷积层4)通道数(深度)为32
             64 * 3 * 3,  # 卷积层6：前一层(卷积层5)通道数(深度)为64

             # 隐藏层：前一层(池化层)，池化层接全连接层需要拉直成一维数组，
             # 因此隐藏层与前一层(池化层)的连接数为池化层的输出节点总数
             64 * 4 * 4,
             # 输出层：前一层(隐藏层)，全连接与前一层全部节点相连，即隐藏层大小
             hidden_size])

        # 权重初始化时的标准差。由于使用ReLU激活函数，因此使用He初始化方式
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        """初始化权重参数和偏置"""
        self.params = {}
        pre_channel_num = input_dim[0]  # 记录上一层的通道数(即滤波器的通道数)
        for idx, conv_param in enumerate([conv_param_1, conv_param_2,
                                          conv_param_3, conv_param_4,
                                          conv_param_5, conv_param_6]):
            # 卷积层滤波器的形状：滤波器个数、通道数、高度、宽度
            self.params['W'+str(idx+1)] = weight_init_scales[idx] *\
                                        np.random.randn(
                                            conv_param['filter_num'],
                                            pre_channel_num,
                                            conv_param['filter_size'],
                                            conv_param['filter_size'])
            self.params['b'+str(idx+1)] = np.zeros(conv_param['filter_num'])

            pre_channel_num = conv_param['filter_num']  # 更新上一层的通道数

        self.params['W7'] = weight_init_scales[6] * np.random.randn(64 * 4 * 4,
                                                                    hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size,
                                                                    output_size)
        self.params['b8'] = np.zeros(output_size)

        """
        构造神经网络：
        书上没有用到之前用的有序字典，其实我觉得很好用，所以就实现了有序字典版本
        Conv1->ReLU1->Conv2->ReLU2->Pool1->
        Conv3->ReLU3->Conv4->ReLU4->Pool2->
        Conv5->ReLU5->Conv6->ReLU6->Pool3->
        Affine1(Hidden Layer1)->ReLU7->Dropout1->
        Affine2(Output Layer1)->Dropout2------->SoftmaxWithLoss
        """
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           stride=conv_param_1['stride'],
                                           pad=conv_param_1['pad'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           stride=conv_param_2['stride'],
                                           pad=conv_param_2['pad'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           stride=conv_param_3['stride'],
                                           pad=conv_param_3['pad'])
        self.layers['ReLU3'] = ReLU()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'],
                                           stride=conv_param_4['stride'],
                                           pad=conv_param_4['pad'])
        self.layers['ReLU4'] = ReLU()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'],
                                           stride=conv_param_5['stride'],
                                           pad=conv_param_5['pad'])
        self.layers['ReLU5'] = ReLU()
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'],
                                           stride=conv_param_6['stride'],
                                           pad=conv_param_6['pad'])
        self.layers['ReLU6'] = ReLU()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['ReLU7'] = ReLU()
        self.layers['Dropout1'] = Dropout(dropout_ratio=0.5)
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers['Dropout2'] = Dropout(dropout_ratio=0.5)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flag=False):
        # 逐层前向传播，预测输入x的输出
        # 如果是Dropout层，需要将train_flag参数传入
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, true_label):
        # 计算损失值。这里只计算了交叉熵误差，也可以加上L2正则化项
        y = self.predict(x, train_flag=True)
        total_loss = self.last_layer.forward(y, true_label)

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
            temp_x = x[i * batch_size: (i + 1) * batch_size]
            temp_true_label = true_label[i * batch_size: (i + 1) * batch_size]
            # 预测这个batch的数据的输出
            y = self.predict(temp_x)
            y = np.argmax(y, axis=1)
            # 统计每个batch的预测正确数
            correct_cnt += np.sum(y == temp_true_label)

        acc = correct_cnt / x.shape[0]  # 计算准确率
        return acc

    def gradient(self, x, true_label):
        # 先前向传播计算中间值
        self.loss(x, true_label)

        # 逐层反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 反向传播结束后从各层提取梯度
        grads = {}
        for i in range(1, 7):
            grads['W'+str(i)] = self.layers['Conv'+str(i)].dW
            grads['b'+str(i)] = self.layers['Conv'+str(i)].db

        grads['W7'] = self.layers['Affine1'].dW
        grads['b7'] = self.layers['Affine1'].db
        grads['W8'] = self.layers['Affine2'].dW
        grads['b8'] = self.layers['Affine2'].db

        return grads

    def save_params(self):
        # 持久化训练好的参数
        file_path = './data/params.pkl'
        params = {}
        for key, val in self.params.items():
            params[key] = val

        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self):
        # 加载参数
        file_path = './data/params.pkl'
        with open(file_path, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i in range(1, 7):
            self.layers['Conv'+str(i)].W = self.params['W'+str(i)]
            self.layers['Conv'+str(i)].b = self.params['b'+str(i)]

        self.layers['Affine1'].W = self.params['W7']
        self.layers['Affine1'].b = self.params['b7']
        self.layers['Affine2'].W = self.params['W8']
        self.layers['Affine2'].b = self.params['b8']
