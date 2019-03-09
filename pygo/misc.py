from enum import IntEnum
from itertools import product
from typing import Tuple, Union

Pass = None
Coord = Tuple[int, int]
Move = Union[Coord, Pass]  # Coord + pass move


def move_to_index(m: Move, size: int) -> int:
    assert m is not Pass, "Pass cannot be converted to index"

    return int(m[0] * size + m[1])


def index_to_coord(i: int, size: int) -> Coord:
    assert 0 <= i < size ** 2, "Illegal index `%d`" % i

    return i // size, i % size


class IllegalMoveError(Exception):
    pass


class Color(IntEnum):
    WHITE = -1
    BLACK = +1

    EMPTY = 0

    def opponent(self):
        if self == Color.WHITE:
            return Color.BLACK

        if self == Color.BLACK:
            return Color.WHITE

        raise ValueError("There is no opponent of %s" % self)


def opposite_color(color: Color) -> Color:
    return color.opponent()


def all_coordinates(size: int):
    return product(range(size), range(size))
