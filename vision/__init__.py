from __future__ import annotations


def match(*args, **kwargs):
    from .matcher import match as _match

    return _match(*args, **kwargs)


def ocr(*args, **kwargs):
    from .ocr import ocr as _ocr

    return _ocr(*args, **kwargs)


def crop_region(*args, **kwargs):
    from .screen import crop_region as _crop_region

    return _crop_region(*args, **kwargs)


def load_image(*args, **kwargs):
    from .screen import load_image as _load_image

    return _load_image(*args, **kwargs)


def screen_matches(*args, **kwargs):
    from .screen import screen_matches as _screen_matches

    return _screen_matches(*args, **kwargs)


def get_ocr_queue_status():
    from .ocr import get_ocr_queue_status as _get_ocr_queue_status

    return _get_ocr_queue_status()

__all__ = ["crop_region", "load_image", "match", "ocr", "screen_matches", "get_ocr_queue_status"]
