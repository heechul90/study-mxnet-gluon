import mxnet as mx
import time
import gluoncv

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

inputShape = (1,3,224,224)

#def gluon_hybridblock(hybridize=True):

class AllOneNet(nn.HybridBlock):
        def __init__(self, verbose=False, **kwargs):
            super(AllOneNet, self).__init__(**kwargs)
            self.verbose = verbose
            # add name_scope on the outer most Sequential
            with self.name_scope():

                layer1=nn.HybridSequential()
                with layer1.name_scope():
                    layer1.add(nn.Conv2D(channels=64,kernel_size=(11,11),strides=4,padding=2,activation="relu"))
                    layer1.add(nn.MaxPool2D(pool_size=(3,3),strides=(2,2)))
                    layer1.add(nn.Conv2D(channels=192,kernel_size=(5,5),strides=1,padding=2,activation="relu"))
                    layer1.add(nn.MaxPool2D(pool_size=(3,3),strides=(2,2)))
                    layer1.add(nn.Conv2D(channels=384,kernel_size=(3,3),strides=1,padding=1,activation="relu"))
                    layer1.add(nn.Conv2D(channels=256,kernel_size=(3,3),strides=1,padding=1,activation="relu"))
                    layer1.add(nn.Conv2D(channels=256,kernel_size=(3,3),strides=1,padding=1,activation="relu"))
                    layer1.add(nn.MaxPool2D(pool_size=(3,3),strides=(2,2)))
                    layer1.add(nn.Flatten())
                    layer1.add(nn.Dense(units=4096,activation="relu"))
                    layer1.add(nn.Dropout(rate=0.5,))
                    layer1.add(nn.Dense(units=4096,activation="relu"))
                    layer1.add(nn.Dropout(rate=0.5,))
                    layer1.add(nn.Dense(units=10,activation="relu"))

                # chain blocks together
                self.net = nn.HybridSequential()
                self.net.add(layer1)

        def hybrid_forward(self,F,x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
            return out

net = AllOneNet(verbose=True)

############### 그래프 ###############
import gluoncv
gluoncv.utils.viz.plot_network(net, shape=inputShape)
#####################################

#    net.cast('float32')
#    net.initialize(mx.init.Xavier(), force_reinit=True, ctx=mx.cpu())
#    net.hybridize()
#    net.forward(mx.nd.empty(inputShape, dtype='float32', ctx=mx.cpu()))
#    filename="model"
#    net.export(filename)


