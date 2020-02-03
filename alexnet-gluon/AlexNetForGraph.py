# AlexNet
# @paper https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
import utils
from mxnet.gluon.block import HybridBlock

ctx = utils.try_gpu()

net = gluon.nn.HybridSequential(prefix='')
with net.name_scope():
    net.add(
        # 第一阶段
        gluon.nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=3, strides=2),
        # 第二阶段
        gluon.nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=3, strides=2),
        # 第三阶段
        gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=3, strides=2),
        # 第四阶段
        gluon.nn.Flatten(),
        gluon.nn.Dense(4096, activation="relu"),
        gluon.nn.Dropout(.5),
        # 第五阶段
        gluon.nn.Dense(4096, activation="relu"),
        gluon.nn.Dropout(.5),
        # 第六阶段
        gluon.nn.Dense(10))

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=224)

net.initialize(ctx=ctx, init=init.Xavier())

###### 그래프 #####
import gluoncv
gluoncv.utils.viz.plot_network(net)





loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)