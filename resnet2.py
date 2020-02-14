import mxnet as mx
import time
import gluoncv

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

inputShape = (1,3,48,48)

#def gluon_hybridblock(hybridize=True):

class block2HybridBlock(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(block2HybridBlock, self).__init__(**kwargs)
            self.path11=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1,in_channels=3,activation="relu")
            self.path12=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path13=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1,in_channels=3,activation="relu")
            self.path14=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path21=nn.Activation(activation="relu")

        def hybrid_forward(self, F, x):
            p1=self.path14(self.path13(self.path12(self.path11(x))))
            p2=self.path21(x)

            if isinstance(p1, mx.sym.Symbol):
                out = mx.sym.concat(p1,p2,dim=1)
            else:
                out = nd.concat(p1,p2,dim=1)
            return out

class AllOneNet(nn.HybridBlock):
        def __init__(self, verbose=False, **kwargs):
            super(AllOneNet, self).__init__(**kwargs)
            self.verbose = verbose
            # add name_scope on the outer most Sequential
            with self.name_scope():

                layer1=nn.HybridSequential()
                with layer1.name_scope():
                    layer1.add(nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1,in_channels=3,activation="relu"))
                    layer1.add(nn.MaxPool2D(pool_size=(2,2),strides=(2,2)))

                layer2=nn.HybridSequential()
                with layer2.name_scope():
                    layer2.add(block2HybridBlock())

                layer3=nn.HybridSequential()
                with layer3.name_scope():
                    layer3.add(nn.Activation(activation="relu"))
                    layer3.add(nn.Dense(units=1,activation="relu"))

                # chain blocks together
                self.net = nn.HybridSequential()
                self.net.add(layer1,layer2,layer3)

        def hybrid_forward(self,F,x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
            return out
net = AllOneNet(verbose=True)
gluoncv.utils.viz.plot_network(net, save_prefix=False)

