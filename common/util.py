# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 17:39
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : util.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def smooth_curve(x):
    """
    用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    使用numpy.random中的permutation函数来实现数据集的打乱
    效果和我之前自己实现的get_data是一样的，不过这个函数是直接打乱数据集，而我是
    不打乱数据集，而是打乱下标。感觉我的做法效率应该更高，不过实现会比较麻烦
    :param x: 特征
    :param t: 标签
    :return:
    """
    permutation = np.random.permutation(x.shape[0])  # 打乱下标
    # 按打乱后的下标调整数据集
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将一个batch的图像数据(4维数据：batch_size * channel * height * width)转换成
    一个2维的矩阵，使得每一行代表一次卷积要用到的元素列表。这样做可以使得内存
    连续，将有助于加速运算。

    通常这样做会导致展开后元素个数比原来多，占用更多内存，但是可以加速运算
    :param input_data: 由(数据量, 通道, 高, 宽)的4维数组构成的输入数据
    :param filter_h: 滤波器的高
    :param filter_w: 滤波器的宽
    :param stride: 步幅
    :param pad: 填充大小(需要在长和高方向都填充，例如填充后h -> h + 2 * pad)
    :return: 转换后的矩阵
    """
    N, C, H, W = input_data.shape  # 数据量, 通道, 高, 宽
    # 卷积后的高。注意这里的除法需要向下取整
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    """
    第一个参数是待填充数组
    第二个参数是填充的形状，（2，3）表示前面填充两个，后面填充三个
    第三个参数是填充的方法：constant连续一样的值填充，有关于其填充值的参数。
    constant_values=（x, y）时前面用x填充，后面用y填充。缺省值x = y = 0

    通过对input_data进行填充，第一个维度数据量N和第二个维度通道C不填充，
    高度H和宽度W分别在前面填充pad个，后面填充pad个。
    填充方法为constant，且填充值为0
    因此填充后img形状为： N * C * (H+2pad) * (W+2pad)
    """
    img = np.pad(input_data,
                 [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')
    """
    生成一个全0的6维数组col
    注意这里前2个维度是N, C固定不动
    中间2个维度是滤波器的高、宽，最后2个维度是输出的高和宽
    
    可以理解为用输出矩阵按照步长扫一遍原多维数组img
    扫完一遍之后可以得到滤波器大小(高x宽)个矩阵，每个矩阵大小就是输出矩阵的大小
    
    然后从上述得到的矩阵中把相同下标的元素挑出来组合成一个新的矩阵，对这个新的
    矩阵用滤波器卷积就可以得到输出矩阵中对应下标的元素。
    
    总结：
    1. 获得 (filter_h * filter_w) 个跟大小为 (out_h * out_w) 的矩阵
    2. 把这些矩阵中下标(i, j)相同的元素组合成一个新的矩阵W_ij
    那么这个新的矩阵W_ij和滤波器做卷积就可以得到输出矩阵中对应下标的元素Oij
    """
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    """
    取filter_h * filter_w 个 out_h * out_w的矩阵
    两层循环结束后得到了滤波器每一个元素各自将会扫到的元素的矩阵，
    然后只需取出滤波器元素扫出的矩阵中相同下标的元素，
    拼接起来就可以得到每次卷积将要用到的元素
    """
    for y in range(filter_h):
        # 按步长从高度方向取out_h个，起点为[0, 1, 2, ..., filter_h - 1]
        y_max = y + stride * out_h
        for x in range(filter_w):
            # 按步长从宽度方向取out_w个，起点为[0, 1, 2, ..., filter_w - 1]
            x_max = x + stride * out_w
            # 滤波器[y, x]下标的元素将会扫到的矩阵
            # img中按步长取out_h行，out_w列
            col[:, :, y, x, :, :] =\
                img[:, :, y: y_max: stride, x: x_max: stride]

    """
    调整col的下标，改成 N * out_h * out_w * C * filter_h * filter_w
    这样调整的原因是 N 个 输出矩阵(out_h * out_w)，
    输出矩阵中一个元素有C个通道，且这C个通道连续存放，
    一个通道的值需要对 filter_h * filter_w 个元素做卷积
    
    转置的做法相当于从上面的for循环获得的filter_h * filter_w 个
    out_h * out_w的矩阵中，取出下标相同的元素，得到filter_h * filter_w 个元素，
    然后组合成一个filter_h * filter_w的矩阵，也就是说，
    col[0][0]代表要得到输出矩阵中下标为0,0的元素，所需要卷积的矩阵
    col[y][x]代表：输出矩阵第y行第x列元素，通过对col[y][x]所代表的矩阵卷积得到
    """
    col = col.transpose((0, 4, 5, 1, 2, 3))

    # 最后reshape成：
    # 一行代表一次卷积要用到的原矩阵的元素列表
    col = col.reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    按照im2col的方法逆向推导回去即可
    :param col: 对应im2col的输出
    :param input_shape: 对应im2col的input_data.shape
    :param filter_h:
    :param filter_w:
    :param stride:
    :param pad:
    :return:
    """
    N, C, H, W = input_shape  # 获取输入形状
    # 计算输出的形状
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # 把im2col最后的变形还原
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        (0, 3, 4, 5, 1, 2))

    # 初始化img为全0的多维数组。
    # 书上是(N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1)，
    # 不影响结果，因为最后会截断之后再返回，但是为做到和im2col逆向，还是去掉较好
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # 因为im2col中是：
            # col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 那么col2im只需把赋值对调即可。书上不是用赋值符号=，而是+=，有误
            img[:, :, y: y_max: stride, x: x_max: stride] =\
                col[:, :, y, x, :, :]

    # 返回图像数据，注意是一个4维数据，需要把填充的部分截断
    return img[:, :, pad: H + pad, pad: W + pad]
