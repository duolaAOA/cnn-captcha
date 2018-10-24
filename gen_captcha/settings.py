#!/usr/bin/env python3
# encoding=utf-8

import os
import sys

import tensorflow as tf

sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 2)[0])
from utils.utils import FileDirHelper

settings = {
    "width": 160,
    "height": 60,
    "max_captcha": 4,
    "char_set_len": 62,
    "image_path": "/data/captcha/image/train/",
    "test_captcha_file_path": "/data/captcha/image/test/",
    # "model_save_path": "/data/captcha/models/",
    "model_save_path": "/data/captcha/zoomeye/models/",
    "predict_img_path": "/data/captcha/predict/predict_test_image.txt"
}

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, settings["height"] * settings["width"]])
Y = tf.placeholder(tf.float32,
                   [None, settings["max_captcha"] * settings["char_set_len"]])

FileDirHelper.make_dir(settings["image_path"])
FileDirHelper.make_dir(settings["test_captcha_file_path"])
FileDirHelper.make_dir(settings["model_save_path"])
