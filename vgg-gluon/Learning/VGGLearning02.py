# VGG 
# @paper https://arxiv.org/abs/1409.1556

from __future__ import print_function
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import init

import utils
import mxnet as mx
import numpy as np
mx.random.seed(1)

##### cpu, gpu 선택 #####
ctx = mx.cpu()
# ctx = mx.gpu()

###### 전처리 ##############################################
def transformer(data, label):
    data = mx.image.imresize(data, 128, 128)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('dataset/FashionMNIST', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST('dataset/FashionMNIST', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')

# 多个 conv layers 加一个 Pooling
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(
                channels=channels, kernel_size=3, padding=1,
                activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out


# 顺序添加多个 vgg_block
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out


###############################################################
# model and params
num_outputs = 10
architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = nn.Sequential()
# add name_scope on the outermost Sequential
# 8 conv layer + 3 denses = VGG 11
# 13 conv layer + 3 denses = VGG 16
# 16 conv layer + 3 denses = VGG 19
with net.name_scope():
    net.add(
        vgg_stack(architecture), nn.Flatten(), nn.Dense(
            4096, activation="relu"), nn.Dropout(.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(.5),
        nn.Dense(num_outputs))

###############################################################

##### 최적화 #####
net.collect_params().initialize(mx.init.Xavier(), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

##### 오차함수 #####
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


