import _pickle as cPickle
import numpy as np
import os
import tensorflow as tf

CIFAR_DIR = ".\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
    data = cPickle.load(f,encoding='bytes')
    print(type(data))
    print(data.keys())
    print(type(data[b'filenames']))
    print(type(data[b'data']))
    print(type(data[b'batch_label']))
    print(type(data[b'labels']))

    print(data[b'data'].shape)
    print(data[b'data'][0:2])
    print(data[b'filenames'][0:2])
    print(data[b'batch_label'])
    print(data[b'labels'][0:2])
#32 * 32 = 1024 * 3(3 aisle) = 3072
#RR-GG－BB＝３０７２
#labels:标签
#[6 9]分别代表第7类和第10类

image_arr = data[b'data'][100]
image_arr = image_arr.reshape((3, 32, 32)) #3表示3个维度，32 32分别表示图片大小32*32
image_arr = image_arr.transpose((1, 2,0))#图片通道显示顺序为 32 32 3 所以需要交换通道

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

imshow(image_arr)
plt.show()

