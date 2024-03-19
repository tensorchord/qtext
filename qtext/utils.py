from functools import wraps
from time import perf_counter

import numpy as np

from qtext.log import logger


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        logger.debug("%s took %s s", func.__name__, perf_counter() - t0)
        return result

    return wrapper


def msgspec_encode_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise NotImplementedError(f"unknown type {type(obj)} for msgspec encoder")
