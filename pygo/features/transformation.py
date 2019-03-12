from typing import Generator, Tuple

import numpy as np

from pygo.features.utils import Order


class TransformationFunction:
    def __init__(self, f, inv, order):
        self._f = f
        self._inv = inv
        self._order = order

    def __call__(self, x):
        return self._f(x)

    @property
    def inv(self):
        return self._inv

    @property
    def order(self):
        return self._order


def _th_rotate_90(planes):
    return np.rot90(planes, axes=(1, 2), k=1)


def _th_rotate_180(planes):
    return np.rot90(planes, axes=(1, 2), k=2)


def _th_rotate_270(planes):
    return np.rot90(planes, axes=(1, 2), k=3)


def _th_fliplr(planes):
    return planes[:, :, ::-1]


def _th_flipud(planes):
    return planes[:, ::-1, :]


def _th_diagonal_left_down(planes):
    return _th_fliplr(_th_rotate_90(planes))


def _th_diagonal_right_down(planes):
    return _th_flipud(_th_rotate_90(planes))


def _tf_rotate_90(planes):
    return np.rot90(planes, axes=(0, 1), k=1)


def _tf_rotate_180(planes):
    return np.rot90(planes, axes=(0, 1), k=2)


def _tf_rotate_270(planes):
    return np.rot90(planes, axes=(0, 1), k=3)


def _tf_fliplr(planes):
    return np.fliplr(planes)


def _tf_flipud(planes):
    return np.flipud(planes)


def _tf_diagonal_left_down(planes):
    return _tf_fliplr(_tf_rotate_90(planes))


def _tf_diagonal_right_down(planes):
    return _tf_flipud(_tf_rotate_90(planes))


class THTransformation:
    """
    Expects tensors' shapes are (channels, row, col)
    """

    identity = TransformationFunction(f=lambda x: x, inv=lambda x: x, order=Order.TH)
    rotate_90 = TransformationFunction(f=_th_rotate_90, inv=_th_rotate_270, order=Order.TH)
    rotate_180 = TransformationFunction(f=_th_rotate_180, inv=_th_rotate_180, order=Order.TH)
    rotate_270 = TransformationFunction(f=_th_rotate_270, inv=_th_rotate_90, order=Order.TH)
    fliplr = TransformationFunction(f=_th_fliplr, inv=_th_fliplr, order=Order.TH)
    flipud = TransformationFunction(f=_th_flipud, inv=_th_flipud, order=Order.TH)
    diagonal_left_down = TransformationFunction(f=_th_diagonal_left_down, inv=_th_diagonal_left_down, order=Order.TH)
    diagonal_right_down = TransformationFunction(f=_th_diagonal_right_down, inv=_th_diagonal_right_down, order=Order.TH)

    ALL = [identity, rotate_90, rotate_180, fliplr, flipud, diagonal_left_down, diagonal_right_down]


class TFTransformation:
    """
    Expects tensors' shapes are (row, col, channels)
    """

    identity = TransformationFunction(f=lambda x: x, inv=lambda x: x, order=Order.TF)
    rotate_90 = TransformationFunction(f=_tf_rotate_90, inv=_tf_rotate_270, order=Order.TF)
    rotate_180 = TransformationFunction(f=_tf_rotate_180, inv=_tf_rotate_180, order=Order.TF)
    rotate_270 = TransformationFunction(f=_tf_rotate_270, inv=_tf_rotate_90, order=Order.TF)
    fliplr = TransformationFunction(f=_tf_fliplr, inv=_tf_fliplr, order=Order.TF)
    flipud = TransformationFunction(f=_tf_flipud, inv=_tf_flipud, order=Order.TF)
    diagonal_left_down = TransformationFunction(f=_tf_diagonal_left_down, inv=_tf_diagonal_left_down, order=Order.TF)
    diagonal_right_down = TransformationFunction(f=_tf_diagonal_right_down, inv=_tf_diagonal_right_down, order=Order.TF)

    ALL = [identity, rotate_90, rotate_180, fliplr, flipud, diagonal_left_down, diagonal_right_down]


def index_to_onehot(action: int, board_size: int) -> np.ndarray:
    assert 0 <= action < board_size ** 2, "Action index (%d) is out of bound of board size %d" % (action, board_size)

    plane = np.zeros(board_size ** 2)
    plane[action] = 1

    return plane.reshape((board_size, board_size))


def onehot_to_index(plane: np.ndarray, board_size: int) -> int:
    return plane.reshape((board_size ** 2,)).argmax()


def transform_index(action: int, f: TransformationFunction, board_size: int) -> int:
    assert action is not None and action != -1 and action != board_size ** 2, "Pass (%s) cannot be transformed" % action

    action_plane = index_to_onehot(action, board_size)

    if f.order == Order.TH:
        action_plane = f(action_plane.reshape((1, board_size, board_size)))
    else:
        action_plane = f(action_plane.reshape((board_size, board_size, 1)))

    return onehot_to_index(action_plane, board_size)


def transformation_functions(order=Order.TH) -> Generator[TransformationFunction, None, None]:
    transformations = THTransformation.ALL if order == Order.TH else TFTransformation.ALL

    for f in transformations:
        yield f


def identical_transform_function(order=Order.TH) -> TransformationFunction:
    return THTransformation.identity if order == Order.TH else TFTransformation.identity


def random_transform_function(order: Order, random_state=None) -> TransformationFunction:
    random_state = random_state or np.random

    transformations = THTransformation.ALL if order == Order.TH else TFTransformation.ALL

    return random_state.choice(transformations)


def identical_transform_generator(order=Order.TH) -> Generator[TransformationFunction, None, None]:
    f = THTransformation.identity if order == Order.TH else TFTransformation.identity

    while True:
        yield f


def random_transform_generator(order=Order.TH) -> Generator[TransformationFunction, None, None]:
    while True:
        yield random_transform_function(order)


def rotate_features_and_action(features: np.ndarray,
                               action: int,
                               board_size: int,
                               f: TransformationFunction) -> Tuple[np.ndarray, int]:
    features = f(features)

    if action is None or action == -1 or action == board_size ** 2:
        return features, board_size ** 2
    else:
        return features, transform_index(action, f, board_size)
