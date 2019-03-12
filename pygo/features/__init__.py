import numpy as np

import pygo.features.cygo as _cfeature
import pygo.features.pygo as _pfeature
from cygo import State as _CState
from pygo import State as _PState
from pygo.features.utils import one_planes as _ones, zero_planes as _zeros


def _resolve_module(state):
    if isinstance(state, _CState):
        return _cfeature

    if isinstance(state, _PState):
        return _pfeature

    raise TypeError


def _only_pfeature(state):
    if isinstance(state, _PState):
        return _pfeature

    raise NotImplementedError("Not implemented for type `%s`" % type(state))


def board(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).board(state, order, dtype)


def board_i(state, i, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).board_i(state, i, order, dtype)


def history_n(state, n, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).history_n(state, n, order, dtype)


def color(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).color(state, order, dtype)


def color_black(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).color_black(state, order, dtype)


def color_white(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).color_white(state, order, dtype)


def leela_board(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).leela_board(state, order, dtype)


def leela_color(state, order, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).leela_color(state, order, dtype)


def stone_color(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).stone_color(state, order, dtype)


def turns_since(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).turns_since(state, order, dtype)


def liberties(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).liberties(state, order, dtype)


def capture_size(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).capture_size(state, order, dtype)


def self_atari_size(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).self_atari_size(state, order, dtype)


def liberties_after_move(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).liberties_after_move(state, order, dtype)


def ladder_capture(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).ladder_capture(state, order, dtype)


def ladder_escape(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).ladder_escape(state, order, dtype)


def sensibleness(state, order, dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).sensibleness(state, order, dtype)


def ones(state, order, dtype=np.float32):
    return _ones(1, state.board_size, order, dtype)


def zeros(state, order, dtype=np.float32):
    return _zeros(1, state.board_size, order, dtype)
