import re

from typing import Optional

from pygo import Color, Move

_LETTERS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # excluding 'i'
_MATCHER = re.compile(r"(?P<alphabet>([A-HJ-Z]))(?P<numeric>([\d]{1,2}))", re.IGNORECASE)


def parse_color(color_str: str) -> Color:
    """
    Parses a given string and returns a formatted string
    as per the result of parsing if the string is
    valid GTP string.
    If not valid, ValueError is thrown.

    :param color_str: string to be parsed
    :return: either `COLOR_BLACK` or `COLOR_WHITE`:
    :exception ValueError if `color_str` is an invalid gtp color string
    """
    if color_str.lower() == 'b':
        return Color.BLACK

    if color_str.lower() == 'w':
        return Color.WHITE

    raise ValueError("Cannot parse string `%s`" % color_str)


def parse_move(move_str: str) -> Move:
    """
    Parses a given string and returns Tuple of int or None as per the given string.
    If the string is not valid, ValueError is thrown.
    """
    if move_str.lower() == 'pass':
        return None

    matched = _MATCHER.match(move_str)
    if matched:
        alphabet = matched.group('alphabet').upper()
        number = matched.group('numeric')

        row = _LETTERS.find(alphabet)
        col = int(number) - 1

        if 0 <= row < 19 and 0 <= col < 19:
            return row, col

    raise ValueError("Cannot parse string `%s`" % move_str)


def move_to_str(move: Move) -> Optional[str]:
    """
    Returns a GTP string as per the given Move instance.
    e.g., "PASS", "A19", "J1" if the argument is None, (0, 18), (8, 0) respectively.
    """
    if move is None:
        return "PASS"

    if not isinstance(move, tuple):
        raise ValueError("Only tuples are acceptable")

    if not (0 <= move[0] < len(_LETTERS) and 0 <= move[1] < len(_LETTERS)):
        raise ValueError("Invalid move `%s`" % (move,))

    row = _LETTERS[move[0]]
    col = move[1] + 1

    return "%s%s" % (row, col)
