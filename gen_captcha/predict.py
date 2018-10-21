#!/usr/bin/env python3
# encoding=utf-8

import time

import numpy as np
from PIL import Image
import tensorflow as tf

import settings
from gen_model import CaptchaModel
from img_handle import ImageHandler
from utils.utils import FileDirHelper
from settings import settings as arg_settings

CHAR_SET_LEN = arg_settings["char_set_len"]
MAX_CAPTCHA = arg_settings["max_captcha"]


def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(
        predict,
        feed_dict={
            settings.X: [captcha_image],
            settings.keep_prob: 1
        })

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return ImageHandler.vec2text(vector)


class HackCaptcha:
    def __init__(self):
        self.model = CaptchaModel()
        FileDirHelper.make_file(arg_settings["predict_img_path"])

    @staticmethod
    def _hack_wrap(sess, predict, captcha_image):

        text_list = sess.run(
            predict,
            feed_dict={
                settings.X: [captcha_image],
                settings.keep_prob: 1
            })
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return ImageHandler.vec2text(vector)

    @staticmethod
    def save_predict_res(line):

        with open(
                arg_settings["predict_img_path"], 'a+',
                encoding='utf-8') as writer:
            writer.write(line + "\n")

    def hack(self):
        output = self.model.create_model()
        predict = tf.argmax(
            tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            checkpoint = tf.train.get_checkpoint_state(
                arg_settings["model_save_path"])
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(
                    sess,
                    tf.train.latest_checkpoint(
                        arg_settings["model_save_path"]))
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                raise FileNotFoundError("Could not find old models")

            stime = time.time()
            right_count = 0
            wrong_count = 0
            for text, img in ImageHandler.get_captcha_code_and_image():
                img = ImageHandler.convert2gray(img)
                img = img.flatten() / 255
                predict_text = self._hack_wrap(sess, predict, img)
                if text == predict_text:
                    right_count += 1
                    print(f"标记: {text}  成功预测: {predict_text}")
                    self.save_predict_res(f"标记: {text}  成功预测: {predict_text}")
                else:
                    wrong_count += 1
                    print(f"标记: {text}  预测: {predict_text}")
                    self.save_predict_res(f"标记: {text}  预测失败: {predict_text}")

            res = 'task:', wrong_count, ' cost time:', (
                time.time() - stime), 's'
            res_count = 'right/total-----', right_count, '/', wrong_count
            print(res)
            print('right/total-----', right_count, '/', wrong_count)
            self.save_predict_res(res)
            self.save_predict_res(str(res_count))

    def _hack_test_captcha(self):
        img_lst = FileDirHelper.get_captcha_list('../test_captcha')
        output = self.model.create_model()
        predict = tf.argmax(
            tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            checkpoint = tf.train.get_checkpoint_state(arg_settings["model_save_path"])
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(
                    sess,tf.train.latest_checkpoint( arg_settings["model_save_path"]))
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                raise FileNotFoundError("Could not find old models")

            for img in img_lst:
                im = Image.open(img)
                im.show()
                captcha_code = img.split("-")[1].split(".")[0]

                captcha_img = Image.open(img)
                captcha_img = np.array(captcha_img)

                img = ImageHandler.convert2gray(captcha_img)
                img = img.flatten() / 255
                predict_text = self._hack_wrap(sess, predict, img)
                if captcha_code == predict_text:
                    print(f"标记: {captcha_code}  成功预测: {predict_text}")
                    self.save_predict_res(f"标记: {captcha_code}  成功预测: {predict_text}")
                else:
                    print(f"标记: {captcha_code}  预测: {predict_text}")


if __name__ == '__main__':
    hack = HackCaptcha()
    hack.hack()
    # hack._hack_test_captcha()
