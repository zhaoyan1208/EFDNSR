import tensorflow as tf
import numpy as np
import random
class Fire(tf.keras.Model):
    def __init__(self,filters_1,filters_2):
        super().__init__()
        self.squeeze = tf.keras.layers.Conv2D(filters = filters_1,kernel_size = 1,strides = 1,padding='SAME',activation=tf.nn.relu)
        self.expand_1 = tf.keras.layers.Conv2D(filters = filters_2,kernel_size = 1,strides = 1,padding='SAME',activation=tf.nn.relu)
        self.expand_3 = tf.keras.layers.Conv2D(filters = filters_2,kernel_size = 3,strides = 1,padding='SAME',activation=tf.nn.relu)
    def call(self,inputs):
        x = self.squeeze(inputs)
        x1 = self.expand_1(x)
        x2 = self.expand_3(x)
        return tf.concat([x1, x2], axis=3)
class Squeezenet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='SAME', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=1000, kernel_size=1, strides=1, padding='SAME',activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=13, strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    def call(self,inputs):
        print(inputs.shape)
        x = self.conv1(inputs)
        print(x.shape)
        fir1 = Fire(16,64)
        x = fir1(x)
        print(x.shape)
        fir2 = Fire(16,64)
        x1 = fir2(x)
        fir3 = Fire(32,128)
        x = fir3(x1)
        x = self.pool1(x)
        fir4 = Fire(32,128)
        x = fir4(x)
        fire5 = Fire(48,192)
        x = fire5(x)
        fir6 = Fire(48,192)
        x = fir6(x)
        fir7 = Fire(64,256)
        x = fir7(x)
        x = self.pool1(x)
        fir8 = Fire(64,256)
        x = fir8(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        out = self.fc1(x)
        return out
z = tf.random.normal([16,224,224,3])
s = Squeezenet()
print(s(z).shape)