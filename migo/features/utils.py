from enum import Enum
from typing import List

import numpy as np


class Order(Enum):
    """CHW v.s. HWC"""
    CHW = 'CHW'                  # theano
    HWC = 'HWC'                  # tensorflow

    @staticmethod
    def tf_to_th(planes: np.ndarray) -> np.ndarray:
        return np.swapaxes(np.swapaxes(planes, 1, 2), 0, 1)

    @staticmethod
    def th_to_tf(planes: np.ndarray) -> np.ndarray:
        return np.swapaxes(np.swapaxes(planes, 0, 1), 1, 2)


def concatenate_along(ordering: Order, arrays: List[np.ndarray]) -> np.ndarray:
    assert all(len(array.shape) == 3 for array in arrays)

    if ordering == Order.CHW:
        return np.concatenate(arrays, axis=0)
    else:
        return np.concatenate(arrays, axis=2)


def empty_planes(channels: int, size: int, order: Order, dtype=np.float32
                 ) -> np.ndarray:
    if order == Order.CHW:
        return np.empty((channels, size, size), dtype=dtype)
    else:
        return np.empty((size, size, channels), dtype=dtype)


def one_planes(channels: int, size: int, order: Order, dtype=np.float32
               ) -> np.ndarray:
    if order == Order.CHW:
        return np.ones((channels, size, size), dtype=dtype)
    else:
        return np.ones((size, size, channels), dtype=dtype)


def zero_planes(channels: int, size: int, order: Order, dtype=np.float32
                ) -> np.ndarray:
    if order == Order.CHW:
        return np.zeros((channels, size, size), dtype=dtype)
    else:
        return np.zeros((size, size, channels), dtype=dtype)
