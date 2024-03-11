from time import perf_counter

from qtext.log import logger


def time_it(func):
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        logger.debug("%s took %s s", func.__name__, perf_counter() - t0)
        return result

    return wrapper
