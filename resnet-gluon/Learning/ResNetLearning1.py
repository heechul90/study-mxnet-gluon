# ResNet
# @paper https://arxiv.org/abs/1512.03385

# mxnet 中有实现 mxnet/gluon/model_zoo/vision/resnet.py

from __future__ import print_function
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import init

import utils
import mxnet as mx
import numpy as np
mx.random.seed(1)

##### 파라미터 ######
# Input_data	(224, 224)
# Batch_size	64
# 초깃값	mx.init.Xavier()
# 경사하강법	sgd
# 학습률	0.01
# Loss	SoftmaxCrossEntropy

##### Test Acc #####
# Epoch0	0.953425481
# Epoch1	0.979567308
# Epoch2	0.985777244
# Epoch3	0.968950321
# Epoch4	0.971754808
# Epoch5	0.920673077
# Epoch6	0.956029647
# Epoch7	0.982371795
# Epoch8	0.991085737
# Epoch9	0.990785256



##### cpu, gpu 선택 #####
ctx = mx.cpu()
# ctx = mx.gpu()


##### 전처리 ##############################################
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/data', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/data', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')


##### ResNet #####
class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2  ## ??
        self.conv1 = nn.Conv2D(
            channels, kernel_size=3, padding=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)


# ResNet 18
class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.HybridSequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2), Residual(64),
                Residual(64))
            # block 3
            b3 = nn.HybridSequential()
            b3.add(Residual(128, same_shape=False), Residual(128))
            # block 4
            b4 = nn.HybridSequential()
            b4.add(Residual(256, same_shape=False), Residual(256))
            # block 5
            b5 = nn.HybridSequential()
            b5.add(Residual(512, same_shape=False), Residual(512))
            # block 6
            b6 = nn.HybridSequential()
            b6.add(nn.AvgPool2D(pool_size=3), nn.Dense(num_classes))
            # chain all blocks together
            self.net = nn.HybridSequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


#############################################################################

##### train #####
net = ResNet(10, verbose=False)

net.collect_params().initialize(mx.init.Xavier(), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})



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
