# GoogLeNet V1
# @paper https://arxiv.org/abs/1409.4842

from __future__ import print_function
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import init

import utils
import mxnet as mx
import numpy as np
mx.random.seed(1)
##### 전처리 ##############################################
ctx = mx.cpu()

def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 128
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/data', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/data', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')

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
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')
        # path 4
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)


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
                nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2))

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
            b6.add(
                nn.Flatten(),
                nn.Dense(num_classes))

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

net = GoogLeNet(10)

net.collect_params().initialize(mx.init.Normal(sigma = 0.05), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})



# 오차 함수
softmax_cross_entropy  = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds = predictions, labels = label)
    return acc.get()


####### train ######
epochs = 10
smoothing_constant = 0.01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ############
        # keep a moving average of the losses
        ############

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))





##### 파라미터 #####
# Input_data	(224, 224)
# Batch_size	128
# 초깃값	        mx.init.Xavier(magnitude=0)
# 경사하강법	    sgd
# 학습률	        0.1
# Loss	        SoftmaxCrossEntropy

##### test_data 정확도 #####
# Epoch0  0.113481
# Epoch1  0.113381
# Epoch2  0.113581
