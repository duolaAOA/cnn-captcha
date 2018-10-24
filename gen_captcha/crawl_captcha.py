#!/usr/bin/env python3
# encoding=utf-8

import os
import sys
from time import time

import requests
from PIL import Image

sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 2)[0])
from utils.utils import FileDirHelper

print(os.getcwd())
os.chdir("../test_captcha")
MAX_CAPTCHA_NUM = 1000


def resize_img(file_dir: str = ".",
               width: int = 160,
               height: int = 60,
               formatt: str = "gif"):
    for img in FileDirHelper.get_captcha_list(file_dir):

        new_img = Image.open(img)
        out = new_img.resize((width, height),
                             Image.ANTIALIAS)  # resize image with high-quality
        out.save(img, formatt)


def download():
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit"
        "/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36"
    }

    for i in range(MAX_CAPTCHA_NUM):
        t = str(int(time() * 1000))
        url = f"http://139.199.188.178:8080/captcha/?t={t}"
        image = requests.get(url=url, headers=headers).content
        print(f"正在下载第{i}张图片")
        with open(str(i) + ".gif", "wb") as f:
            f.write(image)
    resize_img()


# download()
