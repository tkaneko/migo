from collections import OrderedDict
from enum import Enum, auto
from typing import Generator, Tuple

import numpy as np

from migo.features.utils import Order


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


class Dihedral8(Enum):
    IDENTICAL = auto()
    ROTATE_90 = auto()
    ROTATE_180 = auto()
    ROTATE_270 = auto()
    FLIP_UD = auto()
    FLIP_LR = auto()
    DIAG_LEFT_DOWN = auto()
    DIAG_RIGHT_DOWN = auto()

    def inverse(self):
        if self == Dihedral8.ROTATE_90:
            return Dihedral8.ROTATE_270
        if self == Dihedral8.ROTATE_180:
            return Dihedral8.ROTATE_180
        if self == Dihedral8.ROTATE_270:
            return Dihedral8.ROTATE_90

        return self


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

    identity = TransformationFunction(f=lambda x: x, inv=lambda x: x, order=Order.CHW)
    rotate_90 = TransformationFunction(f=_th_rotate_90, inv=_th_rotate_270, order=Order.CHW)
    rotate_180 = TransformationFunction(f=_th_rotate_180, inv=_th_rotate_180, order=Order.CHW)
    rotate_270 = TransformationFunction(f=_th_rotate_270, inv=_th_rotate_90, order=Order.CHW)
    fliplr = TransformationFunction(f=_th_fliplr, inv=_th_fliplr, order=Order.CHW)
    flipud = TransformationFunction(f=_th_flipud, inv=_th_flipud, order=Order.CHW)
    diagonal_left_down = TransformationFunction(f=_th_diagonal_left_down, inv=_th_diagonal_left_down, order=Order.CHW)
    diagonal_right_down = TransformationFunction(f=_th_diagonal_right_down, inv=_th_diagonal_right_down, order=Order.CHW)

    ALL = [identity, rotate_90, rotate_180, fliplr, flipud, diagonal_left_down, diagonal_right_down]


TH_TRANSFORMATIONS = OrderedDict({
    Dihedral8.IDENTICAL: THTransformation.identity,
    Dihedral8.ROTATE_90: THTransformation.rotate_90,
    Dihedral8.ROTATE_180: THTransformation.rotate_180,
    Dihedral8.ROTATE_270: THTransformation.rotate_270,
    Dihedral8.FLIP_UD: THTransformation.flipud,
    Dihedral8.FLIP_LR: THTransformation.fliplr,
    Dihedral8.DIAG_LEFT_DOWN: THTransformation.diagonal_left_down,
    Dihedral8.DIAG_RIGHT_DOWN: THTransformation.diagonal_right_down
})


class TFTransformation:
    """
    Expects tensors' shapes are (row, col, channels)
    """

    identity = TransformationFunction(f=lambda x: x, inv=lambda x: x, order=Order.HWC)
    rotate_90 = TransformationFunction(f=_tf_rotate_90, inv=_tf_rotate_270, order=Order.HWC)
    rotate_180 = TransformationFunction(f=_tf_rotate_180, inv=_tf_rotate_180, order=Order.HWC)
    rotate_270 = TransformationFunction(f=_tf_rotate_270, inv=_tf_rotate_90, order=Order.HWC)
    fliplr = TransformationFunction(f=_tf_fliplr, inv=_tf_fliplr, order=Order.HWC)
    flipud = TransformationFunction(f=_tf_flipud, inv=_tf_flipud, order=Order.HWC)
    diagonal_left_down = TransformationFunction(f=_tf_diagonal_left_down, inv=_tf_diagonal_left_down, order=Order.HWC)
    diagonal_right_down = TransformationFunction(f=_tf_diagonal_right_down, inv=_tf_diagonal_right_down, order=Order.HWC)

    ALL = [identity, rotate_90, rotate_180, fliplr, flipud, diagonal_left_down, diagonal_right_down]


TF_TRANSFORMATIONS = OrderedDict({
    Dihedral8.IDENTICAL: TFTransformation.identity,
    Dihedral8.ROTATE_90: TFTransformation.rotate_90,
    Dihedral8.ROTATE_180: TFTransformation.rotate_180,
    Dihedral8.ROTATE_270: TFTransformation.rotate_270,
    Dihedral8.FLIP_UD: TFTransformation.flipud,
    Dihedral8.FLIP_LR: TFTransformation.fliplr,
    Dihedral8.DIAG_LEFT_DOWN: TFTransformation.diagonal_left_down,
    Dihedral8.DIAG_RIGHT_DOWN: TFTransformation.diagonal_right_down
})


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

    if f.order == Order.CHW:
        action_plane = f(action_plane.reshape((1, board_size, board_size)))
    else:
        action_plane = f(action_plane.reshape((board_size, board_size, 1)))

    return onehot_to_index(action_plane, board_size)


def transformation_functions(order: Order) -> Generator[Tuple[Dihedral8, TransformationFunction], None, None]:
    transformations = TH_TRANSFORMATIONS if order == Order.CHW else TF_TRANSFORMATIONS

    for d, f in transformations.items():
        yield d, f


def identical_transform_function(order: Order) -> Tuple[Dihedral8, TransformationFunction]:
    if order == Order.CHW:
        return Dihedral8.IDENTICAL, THTransformation.identity
    else:
        return Dihedral8.IDENTICAL, TFTransformation.identity


def random_transform_function(order: Order, random_state=None) -> Tuple[Dihedral8, TransformationFunction]:
    random_state = random_state or np.random

    transformations = TH_TRANSFORMATIONS if order == Order.CHW else TF_TRANSFORMATIONS

    dihedral = random_state.choice([Dihedral8.IDENTICAL,
                                    Dihedral8.ROTATE_90,
                                    Dihedral8.ROTATE_180,
                                    Dihedral8.ROTATE_270,
                                    Dihedral8.FLIP_UD,
                                    Dihedral8.FLIP_LR,
                                    Dihedral8.DIAG_LEFT_DOWN,
                                    Dihedral8.DIAG_RIGHT_DOWN])

    return dihedral, transformations[dihedral]


def identical_transform_generator(order: Order) -> Generator[Tuple[Dihedral8, TransformationFunction], None, None]:
    while True:
        yield identical_transform_function(order)


def random_transform_generator(order: Order) -> Generator[Tuple[Dihedral8, TransformationFunction], None, None]:
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
