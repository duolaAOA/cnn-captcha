#!/usr/bin/env python3
# encoding=utf-8


import tensorflow as tf


from gen_model import CaptchaModel
from img_handle import ImageHandler

import settings
from settings import settings as arg_settings


CHAR_SET_LEN = arg_settings["char_set_len"]
MAX_CAPTCHA = arg_settings["max_captcha"]



class TrainCNN:
    def __init__(self):
        self.model = CaptchaModel()


    def train(self):
        output = self.model.create_model()
        # output = self.model.create()
        predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        label = tf.reshape(settings.Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(label, 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)

        with tf.name_scope('my_monitor'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label))

        tf.summary.scalar('my_loss', loss)
        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.name_scope('my_monitor'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('my_accuracy', accuracy)


        saver = tf.train.Saver()  # 将训练过程进行保存
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(
                log_device_placement=False
            )
        )
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = ImageHandler.gen_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={settings.X: batch_x, settings.Y: batch_y, settings.keep_prob: 0.95})
            print(step, 'loss:\t', loss_)

            step += 1

            # 每2000步保存一次实验结果
            if step % 2000 == 0:
                saver.save(sess, arg_settings["model_save_path"], global_step=(step+5))

            # 在测试集上计算精度
            if step % 100 != 0:
                continue

            # 每50 step计算一次准确率，使用新生成的数据
            batch_x_test, batch_y_test = ImageHandler.gen_next_batch(64)  # 新生成的数据集个来做测试
            acc = sess.run(accuracy, feed_dict={settings.X: batch_x_test, settings.Y: batch_y_test, settings.keep_prob: 1.})
            print(step, 'acc---------------------------------\t', acc)

            # 终止条件
            if acc > 0.99:
                break


if __name__ == '__main__':
    print("start")
    train = TrainCNN()
    train.train()
    print('end')
