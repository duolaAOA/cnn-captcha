#!/usr/bin/env python3
# encoding=utf-8

import os
import sys
import json
from random import choice
from string import ascii_lowercase, digits

import numpy as np
import matplotlib.pyplot as plt
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
                self.gen_captcha_code_and_image(i)
                print("train images:" + str(i) + ' images')

        if usage_purpose == "test":
            os.chdir(test_data_dir)
            for i in range(5000):
                self._gen_captcha_code_and_image(i)
                print("test images:" + str(i) + ' images')


def showImage():
    # test
    captcha = Captcha()
    text, image = captcha.gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def main():
    captcha = Captcha()
    captcha.generateMassImages(usage_purpose="train")
    captcha.generateMassImages(usage_purpose="test")
