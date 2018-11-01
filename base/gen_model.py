#!/usr/bin/env python3
# encoding=utf-8

import tensorflow as tf

import settings
from settings import settings as arg_settings


class CaptchaModel:
    def __init__(self):
        self.width = arg_settings["width"]
        self.height = arg_settings["height"]
        self.max_captcha = arg_settings["max_captcha"]
        self.char_set_len = arg_settings["char_set_len"]

    @staticmethod
    def weight_variable(shape):
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(initial_value=initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def create_model(self):

        w_conv1 = self.weight_variable([3, 3, 1, 32])
        b_conv1 = self.bias_variable([32])

        w_conv2 = self.weight_variable([3, 3, 32, 64])
        b_conv2 = self.bias_variable([64])

        w_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        # 输入层
        x = tf.reshape(settings.X, shape=[-1, self.height, self.width, 1])

        h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x, w_conv1, 1), b_conv1))
        h_pool1 = self.max_pool_2x2(h_conv1)
        drop_1 = tf.nn.dropout(h_pool1, settings.keep_prob)

        h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(drop_1, w_conv2, 1), b_conv2))
        h_pool2 = self.max_pool_2x2(h_conv2)
        drop_2 = tf.nn.dropout(h_pool2, settings.keep_prob)

        h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(drop_2, w_conv3, 1), b_conv3))
        h_pool3 = self.max_pool_2x2(h_conv3)
        drop_3 = tf.nn.dropout(h_pool3, settings.keep_prob)

        # Fully connected layer
        w_fc1 = self.weight_variable([11 * 4 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        dense = tf.reshape(drop_3, [-1, w_fc1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_fc1), b_fc1))
        dense = tf.nn.dropout(dense, settings.keep_prob)

        w_out = self.weight_variable([1024, self.max_captcha * self.char_set_len])

        b_out = self.bias_variable([self.max_captcha * self.char_set_len])

        # 输出层
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out
