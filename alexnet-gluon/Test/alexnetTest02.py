import mxnet as mx
import time
import gluoncv

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

inputShape = (1,3,224,224)

from mxnet.gluon.model_zoo import vision

alexnet = vision.alexnet()
inception = vision.inception_v3()


resnet18v1 = vision.resnet18_v1()
resnet18v2 = vision.resnet18_v2()
squeezenet = vision.squeezenet1_0()
densenet = vision.densenet121()
mobilenet = vision.mobilenet0_5()

############### 그래프 ###############
import gluoncv
gluoncv.utils.viz.plot_network(resnet18v1, shape=inputShape)
#####################################


