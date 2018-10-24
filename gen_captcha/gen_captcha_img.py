#!/usr/bin/env python3
# encoding=utf-8

import os
import sys
import json
from io import BytesIO
from math import ceil
from random import choice
from string import ascii_lowercase, digits

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PImage
from PIL import ImageDraw, ImageFont
from captcha.image import Image, ImageCaptcha

sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 2)[0])
from utils.AES_utils import Base64AESCBC, MyCrypt
from utils.utils import FileDirHelper

alphabets = ascii_lowercase + digits

train_data_dir = FileDirHelper.make_dir("/data/captcha/image/train/")
test_data_dir = FileDirHelper.make_dir("/data/captcha/image/test/")


class Captcha:
    def __init__(self):
        self.img = ImageCaptcha()
        self.base = Base64AESCBC()
        self.mc = MyCrypt()

    def gen_captcha_code(self):
        rnd = self.base.encrypt(
            json.dumps({
                'rnd': ''.join([choice(alphabets) for i in range(4)]),
            }))
        data = json.loads(self.mc.decrypt(rnd))

        return data["rnd"]

    def gen_captcha_text_and_image(self):
        captcha_code = self.gen_captcha_code()

        captcha = self.img.generate(captcha_code)
        captcha_img = Image.open(captcha)
        captcha_img = np.array(captcha_img)
        return captcha_code, captcha_img

    def _gen_captcha_code_and_image(self, index):
        while True:
            code, image = self.gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                self.img.write(code, str(index) + '-' + code + '.jpg')
                return

    def gen_captcha_code_and_image(self):
        while True:
            code, image = self.gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return code, image

    def generateMassImages(self, usage_purpose="train"):
        if usage_purpose == "train":
            os.chdir(train_data_dir)
            for i in range(1000):
                self._gen_captcha_code_and_image(i)
                print("train images:" + str(i) + ' images')

        if usage_purpose == "test":
            os.chdir(test_data_dir)
            for i in range(5000):
                self._gen_captcha_code_and_image(i)
                print("test images:" + str(i) + ' images')


class FreshCaptcha:
    def __init__(self):
        self.img = ImageCaptcha()
        self.base = Base64AESCBC()
        self.mc = MyCrypt()
        # 验证码图片尺寸
        self.img_width = 160
        self.img_height = 60
        # 字体颜色
        self.font_color = [
            'goldenrod', 'teal', 'chocolate', 'darkolivegreen', 'plum'
        ]
        # 背景颜色
        self.backgrounds = [
            'snow', 'seashell', 'oldlace', 'mintcream', 'ivory', 'honeydew',
            'ghostwhite', 'cornsilk'
        ]
        # 字体文件路径
        self.font_path = "./FreeSansBoldOblique.ttf"

    def gen_captcha_code(self):
        rnd = self.base.encrypt(
            json.dumps({
                'rnd': ''.join([choice(alphabets) for i in range(4)]),
            }))
        data = json.loads(self.mc.decrypt(rnd))

        return data["rnd"]

    def _get_font_size(self, code: str):
        """ 字体大小"""
        s1 = int(self.img_height * 1.2)
        s2 = int(self.img_width / len(code))
        return int(min((s1, s2)) + max((s1, s2)) * 0.01)

    def gen_captcha_text_and_image(self):

        img = PImage.new('RGB', (self.img_width, self.img_height),
                         choice(self.backgrounds))
        captcha_code = self.gen_captcha_code()
        # 更具图片大小自动调整字体大小
        font_size = self._get_font_size(captcha_code)
        draw = ImageDraw.Draw(img)

        # 写验证码
        x = 8
        for i in captcha_code:
            # 上下抖动量,字数越多,上下抖动越大
            m = int(len(captcha_code))
            y = 2
            self.font = ImageFont.truetype(
                self.font_path.replace('\\', '/'), font_size + int(ceil(m)))
            draw.text((x, y), i, font=self.font, fill=choice(self.font_color))
            x += font_size * 0.8

        del x
        del draw

        out = BytesIO()
        img.save(out, format="gif")
        out.seek(0)
        captcha_img = PImage.open(out)
        captcha_img = np.array(captcha_img)
        return captcha_code, captcha_img, img

    def _gen_captcha_code_and_image(self, index):
        while True:
            code, image, img = self.gen_captcha_text_and_image()
            if image.shape == (32, 86):
                img.save(open(str(index) + '-' + code + '.gif', 'wb'), 'gif')
                return

    def gen_captcha_code_and_image(self):
        while True:
            code, image, img = self.gen_captcha_text_and_image()
            if image.shape == (32, 86):
                return code, image, img

    def generateMassImages(self, usage_purpose="train"):
        if usage_purpose == "train":
            os.chdir(train_data_dir)
            for i in range(1000):
                self._gen_captcha_code_and_image(i)
                print("train images:" + str(i) + ' images')

        if usage_purpose == "test":
            os.chdir(test_data_dir)
            for i in range(5000):
                self._gen_captcha_code_and_image(i)
                print("test images:" + str(i) + ' images')


def showCaptcha1():
    # test
    captcha = Captcha()
    text, image = captcha.gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def showCaptcha2():
    # test
    captcha = FreshCaptcha()
    text, image = captcha.gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def main():
    captcha = Captcha()
    captcha.generateMassImages(usage_purpose="train")
    # captcha.generateMassImages(usage_purpose="test")


def main2():
    captcha = FreshCaptcha()
    captcha.generateMassImages(usage_purpose="train")
    # captcha.generateMassImages(usage_purpose="test")
