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

    def create_model(self, w_alpha=0.01, b_alpha=0.1):
        x = tf.reshape(settings.X, shape=[-1, self.height, self.width, 1])

        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'),
                b_c1))
        conv1 = tf.nn.max_pool(
            conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, settings.keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(
                    conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, settings.keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(
                    conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(
            conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, settings.keep_prob)

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, settings.keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal(
            [1024, self.max_captcha * self.char_set_len]))
        b_out = tf.Variable(
            b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)

        return out
