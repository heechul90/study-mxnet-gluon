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
import gluoncv
mx.random.seed(1)

##### 파라미터 ######
# Input_data	(224, 224)
# Batch_size	64
# 초깃값	mx.init.Xavier()
# 경사하강법	sgd
# 학습률	0.01
# Loss	SoftmaxCrossEntropy

##### Test Acc #####




##### cpu, gpu 선택 #####
ctx = mx.cpu()
# ctx = mx.gpu()

###### 전처리 ##############################################
def transformer(data, label):
    data = mx.image.imresize(data, 32, 32)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/MNIST', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/MNIST', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')


##### DenseNet #####
def conv_block(channels):
    out = nn.HybridSequential()
    out.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size=3, padding=1))
    return out


# Dense Block
# DenseNet的卷积块使用ResNet改进版本的BN->Relu->Conv。
# 每个卷积的输出通道数被称之为growth_rate，这是因为假设输出为in_channels，
# 而且有layers层，那么输出的通道数就是in_channels+growth_rate*layers。
class DenseBlock(nn.HybridBlock):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x


# Transition Block
# 因为使用拼接的缘故，每经过一次拼接输出通道数可能会激增。
# 为了控制模型复杂度，这里引入一个过渡块，它不仅把输入的长宽减半，
# 同时也使用1×1卷积来改变通道数。
def transition_block(channels):
    out = nn.HybridSequential()
    out.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return out


# body
# 121层的DenseNet
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10


def dense_net():
    net = nn.HybridSequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                net.add(transition_block(channels // 2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes))
    return net


#################################################################
##### 최적화 #####
net = dense_net()

net.collect_params().initialize(mx.init.Xavier(), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'gamma1': 0.9,'learning_rate': 0.01})



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