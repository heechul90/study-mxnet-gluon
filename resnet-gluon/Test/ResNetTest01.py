import mxnet as mx
import time
import gluoncv

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn


#def gluon_hybridblock(hybridize=True):

class block2HybridBlock(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(block2HybridBlock, self).__init__(**kwargs)
            self.path11=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1)
            self.path12=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path13=nn.Activation(activation="relu")
            self.path14=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1)
            self.path15=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)

        def hybrid_forward(self, F, x):
            p1=self.path15(self.path14(self.path13(self.path12(self.path11(x)))))

            if isinstance(p1, mx.sym.Symbol):
                out = mx.sym.concat(p1,dim=1)
            else:
                out = nd.concat(p1,dim=1)
            return out

class block4HybridBlock(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(block4HybridBlock, self).__init__(**kwargs)
            self.path11=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1)
            self.path12=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path13=nn.Activation(activation="relu")
            self.path14=nn.Conv2D(channels=64,kernel_size=(3,3),strides=1,padding=1)
            self.path15=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)

        def hybrid_forward(self, F, x):
            p1=self.path15(self.path14(self.path13(self.path12(self.path11(x)))))

            if isinstance(p1, mx.sym.Symbol):
                out = mx.sym.concat(p1,dim=1)
            else:
                out = nd.concat(p1,dim=1)
            return out

class block6HybridBlock(nn.HybridBlock):
        def __init__(self, **kwargs):
            super(block6HybridBlock, self).__init__(**kwargs)
            self.path11=nn.Conv2D(channels=128,kernel_size=(3,3),strides=2,padding=1)
            self.path12=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path13=nn.Activation(activation="relu")
            self.path14=nn.Conv2D(channels=128,kernel_size=(3,3),strides=1,padding=1)
            self.path15=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)
            self.path21=nn.Conv2D(channels=128,kernel_size=(1,1),strides=2)
            self.path22=nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001)

        def hybrid_forward(self, F, x):
            p1=self.path15(self.path14(self.path13(self.path12(self.path11(x)))))
            p2=self.path22(self.path21(x))

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
                    layer1.add(nn.Conv2D(channels=64,kernel_size=(7,7),strides=2,padding=3))
                    layer1.add(nn.BatchNorm(axis =1,momentum =0.9,epsilon=0.00001))
                    layer1.add(nn.Activation(activation="relu"))
                    layer1.add(nn.MaxPool2D(pool_size=(3,3),strides=(2,2)))

                layer2=nn.HybridSequential()
                with layer2.name_scope():
                    layer2.add(block2HybridBlock())

                layer3=nn.HybridSequential()
                with layer3.name_scope():
                    layer3.add(nn.Activation(activation="relu"))

                layer4=nn.HybridSequential()
                with layer4.name_scope():
                    layer4.add(block4HybridBlock())

                layer5=nn.HybridSequential()
                with layer5.name_scope():
                    layer5.add(nn.Activation(activation="relu"))

                layer6=nn.HybridSequential()
                with layer6.name_scope():
                    layer6.add(block6HybridBlock())

                layer7=nn.HybridSequential()
                with layer7.name_scope():
                    layer7.add(nn.Activation(activation="relu"))

                # chain blocks together
                self.net = nn.HybridSequential()
                self.net.add(layer1,layer2,layer3,layer4,layer5,layer6,layer7)

        def hybrid_forward(self,F,x):
            out = x
            for i, b in enumerate(self.net):
                out = b(out)
            return out

net = AllOneNet(verbose=True)

############### 그래프 ###############
import gluoncv
gluoncv.utils.viz.plot_network(net)
#####################################

#    net.cast('float32')
#    net.initialize(mx.init.Xavier(), force_reinit=True, ctx=mx.cpu())
#    net.hybridize()
#    net.forward(mx.nd.empty(inputShape, dtype='float32', ctx=mx.cpu()))
#    filename="model"
#    net.export(filename)