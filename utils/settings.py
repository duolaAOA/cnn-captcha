#!/usr/bin/env python3
# encoding=utf-8

from hashlib import sha256

settings = {
    # "key": sha256('captcha'.encode()).digest(),
    "key": sha256('testtest'.encode()).digest(),
    "iv": b"\xa0\xfby\xc4\xfd\xcb\xc1Cn\xb27:\xb9~\xdd%"
}
