# ResNet
import mxnet as mx
import time
import gluoncv
import numpy as np

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

##### 그래프 #####

dshape = (128, 3, 224, 224)
# dshape = (128, 3, 224, 224)
nclass = 100

def gluon_hybridblock(n=100, hybridize=True):

    class Residual(nn.HybridBlock):
        def __init__(self, channels, same_shape=True, **kwargs):
            super(Residual, self).__init__(**kwargs)
            self.same_shape = same_shape
            strides = 1 if same_shape else 2  ## ??
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
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
                b2.add(nn.MaxPool2D(pool_size=3, strides=2), Residual(64),
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


    net = ResNet(nclass, verbose=True)

    if hybridize:
        net.hybridize()


    net.collect_params().initialize(mx.init.One(), ctx = mx.cpu())
    gluoncv.utils.viz.plot_network(net)




################################################################
gluon_hybridblock()
