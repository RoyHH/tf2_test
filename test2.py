import os

import tensorflow as tf
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#若x是[1, 32, 32, 3]的腾搜人数据
#卷积核个数：4个，卷积核大小：5*5，步长大小：1，padding不做边界处理，激活函数为ReLu。layer——>[1, 28, 28, 4]
layer = layers.Conv2D(4, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
#卷积核个数：4个，卷积核大小：5*5，步长大小：2，padding保留边界处的卷积结果，激活函数为ReLu。layer——>[1, 32, 32, 4]
layer = layers.Conv2D(4, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
#卷积核个数：4个，卷积核大小：5*5，步长大小：2，padding保留边界处的卷积结果，激活函数为ReLu。layer——>[1, 16, 16, 4]
layer = layers.Conv2D(4, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
#卷积核个数：64个，卷积核大小：3*3，默认步长为1，padding保留边界处的卷积结果，激活函数为ReLu。layer——>[1, 32, 32, 64]
layer = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

#若x是[1, 32, 32, 3]的腾搜人数据
#卷积核个数：4个，卷积核大小：5*5*1，偏置值大小：4
w1 = tf.random.normal([5,5,1,4])
b1 = tf.zeros([4])
#conv1———>[1, 28, 28, 4]
conv1 = tf.nn.conv2d(x, w1, strides=1, padding="valid")
#conv1———>[1, 28, 28, 4]——RuLe——>[1, 28, 28, 4]
conv1 = conv1 + b1
conv1 = tf.nn.relu(conv1)
#卷积核个数：4个，卷积核大小：5*5*4，其中的4是因为conv1是4个channel，偏置值大小：16
w2 = tf.random.normal([5,5,4,16])
b2 = tf.zeros([16])
#conv2———>[1, 14, 14, 16]
conv2 = tf.nn.conv2d(conv1, w2, strides=2, padding="same")
... ...


layer = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
layer = layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='same')

#若x是[1, 32, 32, 3]的腾搜人数据
layer = layers.UpSampling2D(size=3)
#out———>[1, 96, 96, 4]
out = layer(x)

layer = layers.BatchNormalization()