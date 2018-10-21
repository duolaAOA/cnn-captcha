#!/usr/bin/env python3
# encoding=utf-8

import base64

from Crypto import Random
from Crypto.Cipher import AES

from utils.settings import settings


class MyCrypt:
    def __init__(self, key=settings["key"]):
        self.bs = 32
        if len(key) >= 32:
            self.key = key[:32]
        else:
            self.key = self._pad(key)

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        cipher = AES.new(self.key, AES.MODE_CBC, settings["iv"])
        return self._unpad(cipher.decrypt(enc)).decode("utf-8")

    def encrypt_ecb(self, raw):
        raw = self._pad(raw)
        key = self._str2bin(self.key)
        cipher = AES.new(key, AES.MODE_ECB)
        enc = cipher.encrypt(raw)
        return enc

    def decrypt_ecb(self, enc):
        key = self._str2bin(self.key)
        cipher = AES.new(key, AES.MODE_ECB)
        dec = cipher.decrypt(enc)
        dec = self._unpad(dec)
        return dec

    def encrypt_ecb_for_css(self, raw):
        length = len(raw)
        left = length % AES.block_size
        body = raw[0:length - left]
        tail = raw[length - left:]
        cipher = AES.new(self.key, AES.MODE_ECB)
        return '%s%s' % (cipher.encrypt(body), tail)

    def decrypt_ecb_for_css(self, raw):
        length = len(raw)
        left = length % AES.block_size
        body = raw[:length - left]
        cipher = AES.new(self.key, AES.MODE_ECB)
        return '%s%s' % (cipher.decrypt(body), raw[length - left:])

    def _pad(self, s):
        return s + (
            self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    def _unpad(self, s):
        return s[:-ord(s[len(s) - 1:])]

    def _str2bin(self, s):
        assert len(s) % 2 == 0
        t = []
        for i in range(0, len(s), 2):
            j = s[i] + s[i + 1]
            t.append(chr(int(j, 16)))
        return ''.join(t)


class Base64AESCBC:
    def __init__(self):
        self.key = settings["key"]

    def encrypt(self, raw):
        raw = self._pad_16(raw)
        cipher = AES.new(self.key, AES.MODE_CBC, settings["iv"])
        return base64.b64encode(cipher.encrypt(raw))

    def decrypt(self, iv, enc):
        enc = base64.b64decode(enc)
        cipher = AES.new(self.key, AES.MODE_CBC, iv[:16])
        return self._unpad(cipher.decrypt(enc))

    @staticmethod
    def _pad_16(s):
        lfsize = 16 - len(s) % 16
        return s + lfsize * chr(lfsize)

    def _unpad(self, s):
        return s[:-ord(s[len(s) - 1:])]
