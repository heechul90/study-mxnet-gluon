# GoogLeNet V1
# @paper https://arxiv.org/abs/1409.4842

from mxnet.gluon import nn
from mxnet import nd
from mxnet import gluon
from mxnet import init

import mxnet as mx
import utils

## mxnet 有自己的实现
## mxnet/gluon/model_zoo/vision/inception.py


# Inception block, http://zh.gluon.ai/_images/inception.svg
# 注意，nn.Conv2D 的 dilation 默认是 (1, 1)，
# 输出的体积由下面的公式算出：
# out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
# out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
class Inception(nn.HybridBlock):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path 1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
        # path 2
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.p2_conv_3 = nn.Conv2D(
            n2_3, kernel_size=3, padding=1, activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(
            n3_5, kernel_size=5, padding=2, activation='relu')
        # path 4
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return mx.sym.concat(p1, p2, p3, p4, dim=1)


# 这是简化版的 GoogLeNet，论文里使用了多个输出，这里只用一个输出。
class GoogLeNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outer most Sequential
        with self.name_scope():
            # block 1
            b1 = nn.HybridSequential()
            b1.add(
                nn.Conv2D(
                    64, kernel_size=7, strides=2, padding=3,
                    activation='relu'), nn.MaxPool2D(pool_size=3, strides=2))
            # block 2
            b2 = nn.HybridSequential()
            b2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 3
            b3 = nn.HybridSequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 4
            b4 = nn.HybridSequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 5
            b5 = nn.HybridSequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2))
            # block 6
            b6 = nn.HybridSequential()
            b6.add(nn.Flatten(), nn.Dense(num_classes))
            # chain blocks together
            self.net = nn.HybridSequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


####################################################################
# train

train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=96)

ctx = utils.try_gpu()
net = GoogLeNet(10)
net.initialize(ctx=ctx, init=init.Xavier())

############### 그래프 ###############
import gluoncv
gluoncv.utils.viz.plot_network(net)
#####################################


loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)