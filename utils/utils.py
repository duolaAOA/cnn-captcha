#!/usr/bin/env python3
# encoding=utf-8

import os


class FileDirHelper:

    @staticmethod
    def make_dir(file_path) -> str:
        abspath = os.path.abspath(file_path)
        if not os.path.exists(abspath):
            os.makedirs(abspath, exist_ok=True)
        return abspath

    @staticmethod
    def make_file(filename, content=''):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    @classmethod
    def get_captcha_list(cls, captcha_file_path: str, recursion=True) -> list:

        res = []
        for item in os.listdir(captcha_file_path):
            full_path = os.path.join(captcha_file_path, item)
            if os.path.isfile(full_path):
                res.append(full_path)
            if recursion and os.path.isdir(full_path):
                res.extend(cls.get_captcha_list(full_path, recursion=recursion))
        return res
