# -*- coding: utf-8 -*-
# @Time         : 2018-08-05 21:05
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : layers.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from common.functions import *
from common.util import im2col, col2im


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
    """
    Affine层保存了权重矩阵和偏置的值及其对应的梯度，在反向传播完成后可以直接从
    Affine层读取梯度
    """
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

        return dx * dout


class BatchNormalization:
    """
    Batch Normalization层
    通过计算一个batch的均值和方差，然后将输入规范化成均值0，方差1的标准正太分布
    然后将规范化后的数据进行缩放和平移的变换，调整数据的分布，使之在经过激活层
    之前或之后有更广的分布，从而更好的进行学习。

    缩放+平移公式：y = gamma * x + beta
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None,
                 running_var=None):
        self.gamma = gamma  # 伽马值
        self.beta = beta  # 贝塔值
        # 动量，有点类似遗忘比例的作用，前向传播时旧的均值、方差在经过momentum遗
        # 忘部分后，由新的均值、方差补充遗忘的部分
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的均值和方差，每列一个均值、方差
        self.running_mean = running_mean
        self.running_var = running_var

        # 反向传播的时候用到的中间数据
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flag=True):
        self.input_shape = x.shape  # 记录输入的形状，在返回结果的时候需要还原
        if x.ndim != 2:  # 如果是图片这种含有多通道的，先调整形状为2维
            # N个样本，每个样本有C个矩阵(通道)，矩阵高H、宽W
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flag)  # 获取前向传播结果
        return out.reshape(*self.input_shape)  # 还原形状

    def __forward(self, x, train_flag):
        # 如果是第一次运行，则先初始化测试用的均值方差为0
        if self.running_mean is None:
            N, D = x.shape  # N行D列，每个样本有D维特征
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        """训练阶段"""
        if train_flag:
            mu = x.mean(axis=0)  # 计算均值mu，每列一个均值

            # var = mean((x - u)^2)
            # x' = (x - u) / std
            xc = x - mu  # 每个样本都减去均值，是标准化的子步骤
            var = np.mean(xc ** 2, axis=0)  # 计算方差，每列一个方差
            # 由方差计算标准差，为防止出现分母为0，加上一个微量10e-7
            std = np.sqrt(var + 10e-7)
            xn = xc / std  # 转换为标准正态分布的数据xn

            # 将计算结果保存在BN层中
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # 测试用的均值方差计算方式：旧值 * momentum + 新值 * (1 - momentum)
            self.running_mean = (self.momentum * self.running_mean +
                                 (1-self.momentum) * mu)
            self.running_var = (self.momentum * self.running_var +
                                (1-self.momentum) * var)
        # 测试阶段，利用训练阶段计算得到的均值方差将数据x转换成标准正态分布
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)

        # 将标准化之后的数据进行缩放平移：y = x * gamma + beta
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:  # 如果是图片这种含有多通道的，先调整形状为2维
            # N个样本，每个样本有C个矩阵(通道)，矩阵高H、宽W
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)  # 获取反向传播的结果
        dx = dx.reshape(*self.input_shape)  # 还原形状
        return dx

    def __backward(self, dout):
        """
        y = xn * gamma + beta

        偏导分别为：
        dbeta = 1
        dgamma = xn
        dxn = gamma

        因此使用链式法则乘以上游传过来的梯度dout：
        dbeta = 1 * dout
        dgamma = xn * dout
        dxn = gamma * dout

        :param dout: 上游传来的梯度
        :return: 传给下游的梯度（dx，即输入的数据x的梯度）
        """
        # 由于beta相当于b，只进行加法运算，可以直接由上游结果对每一列求和得到
        # 之所以在每一列求和，是因为在前向传播的过程中，beta的一个分量就加到了
        # 同一列的所有元素中，那么在拆分成对逐元素求梯度的时候，同一列的所有元素
        # 都可以求出一个dbeta，全部加在一起就是我们要求的dbeta

        # 一个beta分量对应一个维度，所以将这一列的加起来才是这个beta分量的梯度
        dbeta = dout.sum(axis=0)

        # dgamma = xn * dout
        # gamma和beta一样，都是一个分量对应一个维度，所以也要对列求和
        dgamma = np.sum(self.xn * dout, axis=0)

        # dxn = gamma * dout
        dxn = self.gamma * dout

        # xn = xc / std
        # xc的偏导：dxc = 1 / std ； xc的上游dxn
        # xc的梯度：dxc = dxn * (1/std) = dxn / std
        dxc = dxn / self.std

        # xn = xc / std
        # std的偏导：dstd = -xc / std ^ 2  ； std的上游dxn
        # std的梯度：dstd = dxn * (-xc/std^2) = -dxn * xc / std^2
        dstd = -np.sum(dxn * self.xc / (self.std ** 2), axis=0)

        # std = sqrt(var)
        # var的偏导：0.5 * 1 / sqrt(var) = 0.5 / std  ；var的上游dstd
        # var的梯度：dvar = dstd * 0.5 / std
        dvar = 0.5 * dstd / self.std

        # var = mean(xc^2) = sigma(xc^2) / batch_size
        # xc的偏导：2 * xc / batch_size  ；xc的上游dvar
        # xc的梯度：dxc = dvar * 2 * xc / batch_size；

        # 关于dvar * xc 和xc * dvar，由于dvar是一维数组，因此使用*而不是np.dot的
        # 时候好像顺序不影响
        # 由于前面已经计算过一次dxc，因此这里是把两次计算的dxc相加
        dxc += (2.0 / self.batch_size) * self.xc * dvar

        # xc = x - mu
        # mu的偏导：1 ；  mu的上游dxc
        # mu的梯度： dxc的每一列求和，因为mu也是一个分量就影响x的一列
        dmu = np.sum(dxc, axis=0)

        """x的梯度推导不出来……"""
        # xc = x - mu = x - sigma(x) / batch_size
        # x的偏导： 1 - 1 / batch_size       ；       x的上游dxc
        # x的梯度：dxc - dxc / batch_size
        dx = dxc - dmu / self.batch_size  # 这里是减，换成加效果不好，原因不明

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Dropout:
    """
    Dropout层

    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ration = dropout_ratio  # 屏蔽一个神经元的概率
        self.mask = None

    def forward(self, x, train_flag=True):
        if train_flag:  # 在训练阶段屏蔽神经元
            self.mask = np.random.rand(*x.shape) > self.dropout_ration
            return x * self.mask
        else:  # 在测试阶段，激活所有神经元，但是全部神经元的输出都要有所抑制
            return x * (1.0 - self.dropout_ration)

    def backward(self, dout):
        # 类似ReLu的反向传播：
        # 如果前向传播的时候被屏蔽了，那么在反向传播的时候就停止传播(输出0)
        return dout * self.mask


class Convolution:
    """
    卷积层
    """
    def __init__(self, W, b, stride=1, pad=0):
        # 初始化卷积层的参数
        self.W = W  # 滤波器的参数：滤波器个数、通道数、高度、宽度
        self.b = b  # 偏置：大小为滤波器个数，每个滤波器共用一个偏置
        self.stride = stride  # 卷积运算的步长
        # 填充大小(需要在长和高的两个方向都填充)
        # 例如填充后h -> h + 2 * pad。因为在高度的上面和下面都填充pad个元素
        self.pad = pad

        # 中间数据：用于反向传播
        self.x = None
        self.col = None
        self.col_W = None

        # 权重参数和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        # 读取卷积层滤波器的形状：滤波器个数、通道数、高度、宽度
        FN, C, FH, FW = self.W.shape
        # 读取输入数据x的形状：
        # 样本个数(如果是一个batch，就是batch_size)、通道数、高度、宽度
        N, C, H, W = x.shape

        # 计算卷积输出的高度和宽度(向下取整)：(H + 2pad - FH) / stride + 1
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # 用im2col将输入数据x转换成2维的矩阵
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 将滤波器调整为2维矩阵
        # 注意先将滤波器调整为一行一个，然后转置，变成一列一个，
        # 如果直接reshape成(-1, FN)的话，一列并不是一个滤波器
        col_W = self.W.reshape(FN, -1).T

        # 转换后的输入数据x和reshape后的滤波器可以直接用np.dot高速计算
        # 计算完之后要加上偏置b
        # 结果为 二维矩阵，大小为(N * out_h * out_w, FN)
        out = np.dot(col, col_W) + self.b
        # 最后将计算结果调整回原来的形状：
        # 先reshape成 N * H * W * C，
        # 因为im2col中也是调整成一个输出的三个通道连续输出，要保持一致(H, W, C)
        # 然后转置成 N * C * H * W，因为输入是这样的顺序
        # **在这里要领会到reshape和transpose顺序不能对调**
        out = out.reshape(N, out_h, out_w, -1).transpose((0, 3, 1, 2))

        # 记录前向传播的中间结果，将在反向传播中使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        # 读取卷积层滤波器的形状：滤波器个数、通道数、高度、宽度
        FN, C, FH, FW = self.W.shape
        # 将上游传来的梯度转置成 N * H * W * C,与im2col的顺序一致
        # 然后reshape成FN列的二维矩阵
        dout = dout.transpose((0, 2, 3, 1)).reshape(-1, FN)

        # 偏置b在前向传播的时候是加法，因此直接对列求和，组成偏置b
        self.db = np.sum(dout, axis=0)
        # 前向传播方程为：out = col * col_W
        # col_W的偏导为col，上游是dout
        # col形状为(N * out_h * out_w, C * FH * FW)，
        # dout形状为(N * out_h * out_w, FN)，
        # 因此W的梯度为 dcol_W = col.T * dout

        # **上游传来的貌似都是不动的，因此只能col->col.T**
        dcol_W = np.dot(self.col.T, dout)
        # 由于col_W = W.reshape.T，因此反向传播时先T再reshape即可
        self.dW = dcol_W.T.reshape(FN, C, FH, FW)

        # dcol和dcol_W同理，但是col_W形状为(C * FH * FW, FN)，
        # 因此是dout * col_W.T
        dcol = np.dot(dout, self.col_W.T)

        # 卷积层要传给下游的梯度是dx，而col = im2col(x)，因此调用col2im(dcol)即可
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


# TODO
class Pooling:
    """
    最大池化层
    """
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        # 初始化池化层的高度、宽度、步长、填充个数
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # 前向传播的中间结果，将在反向传播中用到
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape  # 输入数据的个数、通道数、高度、宽度
        # 计算池化层输出的高度和宽度(向下取整)
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        # 把输入数据x转换成二维矩阵，每一行代表一次池化要用到的元素列表
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 由于池化是逐个通道进行的，因此需要将输入数据转换成具有池化层大小个列的
        # 二维矩阵，然后每一行取最大值即可
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 按行取最大值，argmax获取的是下标，用于反向传播
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)  # 按行取最大值
        """
        转换成与输入一样的形状：N, C, H, W，
        N个三维矩阵，每个三维矩阵就是一个图像，
        有C个通道，每个通道就是一个二维矩阵，
        也就是说一个图像就是C个H*W的矩阵，然后有N个图像，就是N*C*H*W
        
        out = out.reshape(N, C, out_h, out_w)
        虽然这样结果也是对的，但是为了保持统一，还是先转换成N,H,W,C再转置
        """
        out = out.reshape(N, out_h, out_w, C).transpose((0, 3, 1, 2))

        # 保存中间结果
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        其实反向传播就是按照前向传播一行行倒推就行了。
        目的就是求出dx，传给下游。如果本层有参数需要训练，
        那么还要将该参数的梯度保存起来
        :param dout:
        :return:
        """
        # 将上游传来的梯度转置成 N * out_h * out_w * C,与im2col的顺序一致
        # 与forward逆向
        dout = dout.transpose((0, 2, 3, 1))

        # forward中求np.max，在backward中就反向求dmax
        pool_size = self.pool_h * self.pool_w  # 一次池化的元素个数
        # 前向传播时展开后的二维矩阵形状是(-1, pool_size)，
        # 在反向传播中也要保持形状，把dout拉直成一维数组后增加pool_size列
        dmax = np.zeros((dout.size, pool_size))
        # 利用前向传播时记录的中间结果(每行最大值的索引)，
        # 将上面生成的二维矩阵中每一行最大值的位置填入dout对应的值
        # dmax其他元素都为0，但是最大值的位置就是dout对应行的最大值。

        # 也就是说，只有最大值所在的位置梯度才能传递下去
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] =\
            dout.flatten()
        # dmax调整为 N * H * W * C * pool_size。
        # 由于forward中将col.reshape(-1, pool_size)，在backward中也要保持形状
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # 因为col的形状是： N * out_h * out_w, -1
        # 所以dcol也要调整为一样的形状
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # 由于col = im2col(x)，因此dx = col2im(col)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride,
                    self.pad)

        return dx
