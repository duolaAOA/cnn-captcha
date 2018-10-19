#!/usr/bin/env python3
# encoding=utf-8

import os

def make_dir(file_path):
    abspath = os.path.abspath(file_path)
    if not os.path.exists(abspath):
        os.makedirs(abspath, exist_ok=True)
    return abspath