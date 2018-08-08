# -*- coding: utf-8 -*-
# @Time         : 2018-08-08 19:55
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : train_neural_network.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np

from chapter3.mnist import load_mnist
from chapter5.two_layer_net import TwoLayerNet


def main():
    """
    和chapter4的train_neural_network一样，都是利用two_layer_net中的TwoLayer类
    构造神经网络，然后对MNIST的数据集进行预测并计算准确率

    但是比chapter4中的实现要方便巧妙得多，因为将每一层设计成一个类，然后在构造
    神经网络的时候就不用管层里面的具体细节，方便构造也方便求梯度
    :return:
    """
    (train_x, train_y), (test_x, test_y) = load_mnist(one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iterations = 100000
    train_size = train_x.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    batch_mask = np.arange(train_size)
    np.random.shuffle(batch_mask)

    left = 0
    epoch_num = 0
    for i in range(int(iterations / batch_size) * batch_size):
        batch_x = train_x[batch_mask[left: left+batch_size]]
        batch_y = train_y[batch_mask[left: left+batch_size]]
        
        grad = network.gradient(batch_x, batch_y)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
            
        loss = network.loss(batch_x, batch_y)
        train_loss_list.append(loss)

        left += batch_size
        if left >= train_size:
            left = 0
            epoch_num += 1
            train_acc = network.accuracy(train_x, train_y)
            test_acc = network.accuracy(test_x, test_y)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("No.%d epoch:" % epoch_num)
            print("train acc: %f\ttest acc: %f" % (train_acc, test_acc))

    return train_acc_list, test_acc_list, train_loss_list


if __name__ == '__main__':
    main()
