# GoogLeNet V1
# @paper https://arxiv.org/abs/1409.4842
from __future__ import print_function
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import init
from glob import glob
from keras.preprocessing import image

import mxnet as mx
import utils
import numpy as np
import cv2, os, random
import matplotlib.pyplot as plt
mx.random.seed(1)


##### 전처리 ##############################################
##### dogs #########################
# path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
# ROW, COL = 96, 96
# dogs = []
# dog_path = os.path.join(path, 'dog.*')
# for dog_img in glob(dog_path):
#     dog = mx.image.imread(dog_img)
#     dog = mx.image.imresize(dog, ROW, COL)
#     dog = mx.nd.transpose(dog.astype('float32'), (2, 0, 1)) / 255
#     dogs.append(dog)
#
# y_dogs = [1 for item in enumerate(dogs)]
# y_dogs = mx.nd.array(y_dogs)
#
# ##### cats #########################
# path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
# ROW, COL = 96, 96
# cats = []
# cat_path = os.path.join(path, 'cat.*')
# for cat_img in glob(cat_path):
#     cat = mx.image.imread(cat_img)
#     cat = mx.image.imresize(cat, ROW, COL)
#     cat = mx.nd.transpose(cat.astype('float32'), (2, 0, 1)) / 255
#     cats.append(cat)
# y_cats = [0 for item in enumerate(cats)]
# y_cats = mx.nd.array(y_cats)
#
# mx.nd.concatenate((dogs, cats), axis = 0)


################## train #########################
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
ROW, COL = 96, 96
dogs = []
dog_path = os.path.join(path, 'dog.*')
for dog_img in glob(dog_path):
    dog = cv2.imread(dog_img)
    dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
    dog = cv2.resize(dog, (ROW, COL))
    dog = image.img_to_array(dog) / 255
    dogs.append(dog)
print('Some dog images starting with 5 loaded')
y_dogs = [1 for item in enumerate(dogs)]


path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
ROW, COL = 96, 96
cats = []
cat_path = os.path.join(path, 'cat.*')
for cat_img in glob(cat_path):
    cat = cv2.imread(cat_img)
    cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
    cat = cv2.resize(cat, (ROW, COL))
    cat = image.img_to_array(cat) / 255
    cats.append(dog)
print('Some cat images starting with 5 loaded')
y_cats = [1 for item in enumerate(cats)]

X = np.concatenate((dogs, cats), axis = 0)
y = np.concatenate((y_dogs, y_cats), axis = 0)
len(X)
len(y)
X = mx.nd.array(X)
y = mx.nd.array(y)

batch_size = 64
train_data = mx.io.NDArrayIter(X, y, batch_size, shuffle=True)

###########################################
################## test #########################
path = 'D:/HeechulFromGithub/dataset/dogs-vs-cats/train/'
ROW, COL = 96, 96
test = []
test_path = os.path.join(path, '*')
for test_img in glob(test_path):
    t = cv2.imread(test_img)
    t = cv2.resize(t, (ROW, COL))
    t = image.img_to_array(t) / 255
    cats.append(t)
print('Some test images starting with 5 loaded')



# data (NDArray) – Source input
# axes (Shape(tuple), optional, default=[]) – Target axis order. By default the axes will be inverted.
# out (NDArray, optional) – The output NDArray to hold the result.

## mxnet 有自己的实现
## mxnet/gluon/model_zoo/vision/inception.py


# Inception block, http://zh.gluon.ai/_images/inception.svg
# 注意，nn.Conv2D 的 dilation 默认是 (1, 1)，
# 输出的体积由下面的公式算出：
# out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
# out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path 1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
        # path 2
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.p2_conv_3 = nn.Conv2D(
            n2_3, kernel_size=3, padding=1, activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(
            n3_5, kernel_size=5, padding=2, activation='relu')
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
class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outer most Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(
                    64, kernel_size=7, strides=2, padding=3,
                    activation='relu'), nn.MaxPool2D(pool_size=3, strides=2))
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 3
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 4
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2))

            # block 5
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2))
            # block 6
            b6 = nn.Sequential()
            b6.add(nn.Flatten(), nn.Dense(num_classes))
            # chain blocks together
            self.net = nn.Sequential()
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
# train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=96)
#
# ctx = utils.try_gpu()
# net = GoogLeNet(10)
# net.initialize(ctx=ctx, init=init.Xavier())
#
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)


####################################################
# 표준편차가 0.05인 정규 분포에서 모델의 파라미터 전체에 대해
net = GoogLeNet(10)

# 임의 값으로 시작
net.collect_params().initialize(mx.init.Normal(sigma=0.05))

# softmax cross entropy loss 함수를 사용하여 # 모델이 정답을 얼마나 잘 예측할 수 있는지 평가하도록 선택
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# SGD(Stochastic Gradient Descent) 학습 알고리즘을 사용하고
# 학습 속도 하이퍼파라미터를 .1로 설정하도록 선택
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


epochs = 10
for e in range(epochs):
    for i, batch in enumerate(train_data):
        data = batch.data[0]
        label = batch.label[0]
        with autograd.record(): # 파생물 기록 시작
            output = net(data) # 순방향 반복
            loss = loss(output, label)
            loss.backward()
        trainer.step(data.shape[0])


acc = mx.metric.Accuracy()# 정확성 지표 초기화
output = net(test_data_mx) # 신경망을 통해 테스트 데이터 실행
predictions = ndarray.argmax(output, axis=1) # 테스트 데이터 예측
acc.update(preds=predictions, labels=test_label_mx) # 정확성 계산
print(acc)