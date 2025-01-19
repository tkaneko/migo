from enum import IntEnum
from itertools import product
from typing import Tuple, Union

Pass = None
Coord = Tuple[int, int]
Move = Union[Coord, Pass]  # Coord + pass move


def move_to_index(m: Move, size: int) -> int:
    """
    >>> migo.misc.move_to_index((1, 2), size=4)
    6
    """
    assert m is not Pass, "Pass cannot be converted to index"

    return int(m[0] * size + m[1])


def index_to_coord(i: int, size: int) -> Coord:
    """
    >>> migo.misc.index_to_coord(6, size=4)
    (1, 2)
    """
    assert 0 <= i < size ** 2, "Illegal index `%d`" % i

    return i // size, i % size


class IllegalMoveError(Exception):
    pass


class Color(IntEnum):
    """define color id"""
    WHITE = -1                  #: white
    BLACK = +1                  #: black

    EMPTY = 0                   #: empty

    def opponent(self):
        """opponent color for black and white, raise exception for empty

        >>> migo.Color.WHITE.opponent() == migo.Color.BLACK
        True
        """
        if self == Color.WHITE:
            return Color.BLACK

        if self == Color.BLACK:
            return Color.WHITE

        raise ValueError("There is no opponent of %s" % self)


def opposite_color(color: Color) -> Color:
    return color.opponent()


def all_coordinates(size: int):
    return product(range(size), range(size))
