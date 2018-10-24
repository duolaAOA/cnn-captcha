#!/usr/bin/env python3
# encoding=utf-8

import tensorflow as tf

import settings
from gen_model import CaptchaModel
from img_handle import ImageHandler
from settings import settings as arg_settings

CHAR_SET_LEN = arg_settings["char_set_len"]
MAX_CAPTCHA = arg_settings["max_captcha"]
ACC_BREAK = 0.998


class TrainCNN:
    def __init__(self):
        self.model = CaptchaModel()

    def train(self):
        output = self.model.create_model()
        predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        label = tf.reshape(settings.Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(label, 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)

        with tf.name_scope('my_monitor'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=predict, labels=label))

        tf.summary.scalar('my_loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.name_scope('my_monitor'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('my_accuracy', accuracy)

        saver = tf.train.Saver()
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            # batch_x, batch_y = ImageHandler.gen_next_batch(64)
            batch_x, batch_y = ImageHandler.gen_fresh_batch(128)
            _, loss_ = sess.run([optimizer, loss],
                                feed_dict={
                                    settings.X: batch_x,
                                    settings.Y: batch_y,
                                    settings.keep_prob: 0.95
                                })
            print(step, 'loss:\t', loss_)

            step += 1

            # 每1000步保存一次实验结果
            if step % 1000 == 0:
                saver.save(
                    sess, arg_settings["model_save_path"], global_step=step)

            # 在测试集上计算精度
            if step % 200 != 0:
                continue

            # batch_x_test, batch_y_test = ImageHandler.gen_next_batch(64)  # 新生成数据集个测试
            batch_x_test, batch_y_test = ImageHandler.gen_fresh_batch(
                128)  # 新生成数据集个测试
            acc = sess.run(
                accuracy,
                feed_dict={
                    settings.X: batch_x_test,
                    settings.Y: batch_y_test,
                    settings.keep_prob: 1.
                })
            print(step, 'acc-------------------\t', acc)

            # 终止条件
            if acc > ACC_BREAK:
                saver.save(
                    sess, arg_settings["model_save_path"], global_step=step)
                break


if __name__ == '__main__':
    print("start")
    train = TrainCNN()
    train.train()
    print('end')
