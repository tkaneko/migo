import numpy as np

import migo.features.cygo as _cfeature
import migo.features.migo as _pfeature
from .cygo import State as _CState
from .migo import State as _PState
from .utils import one_planes as _ones, zero_planes as _zeros, Order


__all__ = [
    'Order', 'board', 'board_i', 'history_n', 'color', 'color_black',
    'color_white', 'leela_board', 'leela_color', 'ones', 'zeros'
]


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


def board(state: _CState | _PState, order: Order = Order.CHW,
          dtype=np.float32) -> np.ndarray:
    """return two planes each for player to move and opponent

    .. warning::
       the order of planes depends on which player to move.

    >>> state = migo.State(3)
    >>> state.make_move((1, 2))
    False
    >>> migo.features.board(state)
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    ...
           [[0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.]]], dtype=float32)
    >>> migo.features.board(state, migo.features.Order.HWC)
    array([[[0., 0.],
            [0., 0.],
            [0., 0.]],
    ...
           [[0., 0.],
            [0., 0.],
            [0., 1.]],
    ...
           [[0., 0.],
            [0., 0.],
            [0., 0.]]], dtype=float32)
    >>> state.make_move((1, 1))
    False
    >>> migo.features.board(state)
    array([[[0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.]],
    ...
           [[0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]]], dtype=float32)
    """
    return _resolve_module(state).board(state, order, dtype)


def board_i(state, i: int, order: Order = Order.CHW, dtype=np.float32
            ) -> np.ndarray:
    """Get the i-th before board feature (both color's).

    :params i: should be in [0, state.max_history_n()] (inclusive)
    """
    return _resolve_module(state).board_i(state, i, order, dtype)


def history_n(state, n, order: Order = Order.CHW, dtype=np.float32
              ) -> np.ndarray:
    """Get the board history from n-th before (both color's features)

    `migo.features.history_n(state, n)[:2]` is equivalent to
    `migo.features.board(state)`

    :return: planes with `C = (n + 1) * 2`
    """
    return _resolve_module(state).history_n(state, n, order, dtype)


def color(state, order: Order = Order.CHW, dtype=np.float32) -> np.ndarray:
    return _resolve_module(state).color(state, order, dtype)


def color_black(state, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    """plane filled with ons or zeros if black to move

    >>> state = migo.State(2)
    >>> migo.features.color_black(state)
    array([[[1., 1.],
            [1., 1.]]], dtype=float32)
    >>> state.make_move((0, 0))
    False
    >>> migo.features.color_black(state)
    array([[[0., 0.],
            [0., 0.]]], dtype=float32)
    """
    return _resolve_module(state).color_black(state, order, dtype)


def color_white(state, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    """plane filled with ones or zeros if white to move"""
    return _resolve_module(state).color_white(state, order, dtype)


def leela_board(state, n: int, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    return _resolve_module(state).leela_board(state, order, n=n, dtype=dtype)


def leela_color(state, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    return _resolve_module(state).leela_color(state, order, dtype)


def stone_color(state, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    """return three planes each for player to move, opponent, and empty

    .. warning::
       the order of planes depends on which player to move.

    >>> state = migo.State(3)
    >>> state.make_move((1, 2))
    False
    >>> migo.features.stone_color(state)
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    ...
           [[0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.]],
    ...
           [[1., 1., 1.],
            [1., 1., 0.],
            [1., 1., 1.]]], dtype=float32)
    >>> migo.features.stone_color(state, migo.features.Order.HWC)
    array([[[0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.]],
    ...
           [[0., 0., 1.],
            [0., 0., 1.],
            [0., 1., 0.]],
    ...
           [[0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.]]], dtype=float32)
    """
    return _only_pfeature(state).stone_color(state, order, dtype)


def turns_since(state: _PState, order: Order = Order.CHW, dtype=np.float32
                ) -> np.ndarray:
    return _only_pfeature(state).turns_since(state, order, dtype)


def liberties(state: _PState, order: Order = Order.CHW, dtype=np.float32
              ) -> np.ndarray:
    return _only_pfeature(state).liberties(state, order, dtype)


def capture_size(state: _PState, order: Order = Order.CHW, dtype=np.float32
                 ) -> np.ndarray:
    return _only_pfeature(state).capture_size(state, order, dtype)


def self_atari_size(state: _PState, order: Order = Order.CHW, dtype=np.float32
                    ) -> np.ndarray:
    return _only_pfeature(state).self_atari_size(state, order, dtype)


def liberties_after_move(state: _PState, order: Order = Order.CHW,
                         dtype=np.float32) -> np.ndarray:
    return _only_pfeature(state).liberties_after_move(state, order, dtype)


def ladder_capture(state: _PState, order: Order = Order.CHW, dtype=np.float32
                   ) -> np.ndarray:
    return _only_pfeature(state).ladder_capture(state, order, dtype)


def ladder_escape(state: _PState, order: Order = Order.CHW, dtype=np.float32
                  ) -> np.ndarray:
    return _only_pfeature(state).ladder_escape(state, order, dtype)


def sensibleness(state: _PState, order: Order = Order.CHW, dtype=np.float32
                 ) -> np.ndarray:
    return _only_pfeature(state).sensibleness(state, order, dtype)


def ones(state: _PState | _CState, order: Order = Order.CHW, dtype=np.float32
         ) -> np.ndarray:
    return _ones(1, state.board_size, order, dtype)


def zeros(state: _PState | _CState, order: Order = Order.CHW, dtype=np.float32
          ) -> np.ndarray:
    return _zeros(1, state.board_size, order, dtype)
