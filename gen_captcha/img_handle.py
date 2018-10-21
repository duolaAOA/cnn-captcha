#!/usr/bin/env python3
# encoding=utf-8

import numpy as np

from captcha.image import Image

from gen_captcha_img import Captcha
from utils.utils import FileDirHelper
from settings import settings as arg_settings

# 验证码图片的存放路径
IMAGE_PATH = arg_settings["image_path"]
# 验证码图片的宽度
IMAGE_WIDTH = arg_settings["width"]

# 验证码图片的高度
IMAGE_HEIGHT = arg_settings["height"]

CHAR_SET_LEN = arg_settings["char_set_len"]
MAX_CAPTCHA = arg_settings["max_captcha"]

test_captcha_file_path = arg_settings["test_captcha_file_path"]

#存放训练好的模型的路径
MODEL_SAVE_PATH = arg_settings["model_save_path"]


class ImageHandler:

    captcha = Captcha()

    @staticmethod
    def convert2gray(img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            return gray
        else:
            return img

    @staticmethod
    def char2pos(c):
        """
        字符验证码，字符串转成位置信息
        :param c:
        :return:
        """
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    @staticmethod
    def pos2char(char_idx: int):
        """
        根据位置信息转化为索引信息
        :param char_idx:
        :return:
        """

        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')

        return chr(char_code)

    @classmethod
    def text2vec(cls, text):
        text_len = len(text)
        if text_len > MAX_CAPTCHA:
            raise ValueError('验证码最长4个字符')

        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

        for i, c in enumerate(text):
            idx = i * CHAR_SET_LEN + cls.char2pos(c)
            vector[idx] = 1
        return vector

    @classmethod
    def vec2text(cls, vec):
        text = []
        char_pos = vec.nonzero()[0]
        for _, c in enumerate(char_pos):
            char_idx = c % CHAR_SET_LEN
            char_code = cls.pos2char(char_idx)
            text.append(char_code)

        return "".join(text)

    @staticmethod
    def get_captcha_code_and_image():
        for captcha in FileDirHelper.get_captcha_list(
                captcha_file_path=test_captcha_file_path, recursion=False):
            captcha_code = captcha.split("-")[1].split(".")[0]
            captcha_img = Image.open(captcha)
            captcha_img = np.array(captcha_img)
            yield captcha_code, captcha_img

    @classmethod
    def gen_next_batch(cls, batch_size=64):
        """
        # 生成一个训练batch
        :param batch_size:
        :return:
        """
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

        for i in range(batch_size):
            text, image = cls.captcha.gen_captcha_text_and_image()
            image = cls.convert2gray(image)

            batch_x[i, :] = image.flatten(
            ) / 255  # (image.flatten()-128)/128  mean为0
            batch_y[i, :] = cls.text2vec(text)

        return batch_x, batch_y
