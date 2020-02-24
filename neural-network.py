#单层神经元结构->扩展为多层神经元结构
import tensorflow as tf
import os
import _pickle as cPickle
import numpy as np

CIFAR_DIR = ".\cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

#函数load_data负责将pickle文件中将数据读取出来
def load_data(filename):
    """read data from data file."""
    with open(filename,'rb') as f:
        data = cPickle.load(f, encoding='bytes')
        return data[b'data'],data[b'labels']#仅需图片的像素值和label值即可

class CifaData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
            """
            for item, label in zip(data, labels):
                if label in [0, 1]:
                    all_data.append(item)
                    all_labels.append(label)
            """
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1 #归一化处理
        self._labels = np.hstack(all_labels)
       # print(self._data.shape)
       # print(self._labels.shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        #混排[0,1,2,3,4,5]->[5,2,3,4,1,0]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size example as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifaData(train_filenames, True)
test_data = CifaData(test_filenames,False)

batch_data, batch_labels = train_data.next_batch(10)
#print(batch_data)
#print(batch_labels)

#先搭建tensorflow的计算图，然后再执行计算图
x = tf.placeholder(tf.float32,[None,3072])#placeholder占位符（data） shape:[None,3072]
# [None], eg:[0,6,5,3]
y = tf.placeholder(tf.int64,[None])#只一个维度
#minibatch梯度下降 None:不确定样本个数（应对batchsize的可变性）

#具有三个隐含层的神经网络，其激活函数都为relu
hidden1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, 50, activation=tf.nn.relu)
y_ = tf.layers.dense(hidden3, 10)


#交叉熵损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#y_ -> softmax
#y = one_hot
#loss = ylogy_

"""
#[None,1] 将y_值变为概率值
p_y_1 = tf.nn.sigmoid(y_)
#[None,1]
y_reshaped = tf.reshape(y,(-1,1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
"""

#bool
#predict = p_y_1 > 0.5

#indices
predict = tf.argmax(y_, 1)
#[1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))#求平均

#定义梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)#running rate 1e-3

#（执行计算图）先初始化变量
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
                 [loss, accuracy , train_op],
                 feed_dict={x: batch_data, y: batch_labels})#x,y分别为cifar10数据
        if (i + 1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f '% (i+1, loss_val, acc_val))
        if (i + 1) % 5000 == 0:
            test_data = CifaData(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                                feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            #print('[Test] step: %d, acc: %4.5f' % ((i+1), test_acc))
            print('[Test] Step: %d, acc: %4.5f' % (i + 1, test_acc))
