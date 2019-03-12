import numpy as np

import cygo.features as F
from cygo import State, Color, opposite_color
from pygo.features.utils import Order


def board(state: State, order=Order.TH, dtype=np.float32) -> np.ndarray:
    return board_i(state, 0, order, dtype=dtype)


def board_i(state: State, i: int, order=Order.TH, dtype=np.float32) -> np.ndarray:
    planes = F.board_i(state, i).astype(dtype)

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def history_n(state: State, n: int, order=Order.TH, dtype=np.float32) -> np.ndarray:
    planes = F.history_n(state, n).astype(dtype)

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def color(state: State, order=Order.TH, dtype=np.float32) -> np.ndarray:
    return color_black(state, order, dtype)


def color_black(state: State, order=Order.TH, dtype=np.float32) -> np.ndarray:
    planes = F.color_black(state).astype(dtype)

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def color_white(state: State, order=Order.TH, dtype=np.float32) -> np.ndarray:
    planes = F.color_white(state).astype(dtype)

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def leela_board(state: State, order=Order.TH, n=7, dtype=np.float32) -> np.ndarray:
    planes = np.concatenate([F.history_n(state, n, state.current_player),
                             F.history_n(state, n, opposite_color(state.current_player))]).astype(dtype)

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def leela_color(state: State, order=Order.TH, dtype=np.float32) -> np.ndarray:
    planes = np.concatenate([color_black(state, dtype=dtype), color_white(state, dtype=dtype)])

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes


def history_n_bw(state: State, n: int, order=Order.TH, dtype=np.float32) -> np.ndarray:
    b = F.history_n(state, n, Color.BLACK).astype(dtype)
    w = F.history_n(state, n, Color.WHITE).astype(dtype)

    planes = np.empty((len(b) + len(w), state.board_size, state.board_size), dtype=dtype)

    planes[0::2] = b
    planes[1::2] = w

    if order == Order.TF:
        planes = Order.tf_to_th(planes)

    return planes
