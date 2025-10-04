import re

from typing import Optional

from migo import Color, Move

_LETTERS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # excluding 'i'
_LETTERS_cgoban = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # including 'i'
_MATCHER = re.compile(r"^(?P<alphabet>([A-HJ-Z]))(?P<numeric>([\d]{1,2}))$",
                      re.IGNORECASE)
_MATCHER_cgoban = re.compile(r"^(?P<alphabet>([A-Z]))(?P<col>([A-Z]))$",
                             re.IGNORECASE)  # cgoban


def parse_color(color_str: str) -> Color:
    """
    Parses a given string and returns a formatted string
    as per the result of parsing if the string is
    valid GTP string.
    If not valid, ValueError is thrown.

    :param color_str: string to be parsed
    :return: either :py:class:`migo.Color.BLACK` or :py:class:`migo.Color.WHITE`
    :exception ValueError: if `color_str` is an invalid gtp color string

    >>> migo.gtp.parse_color('b') == migo.Color.BLACK
    True
    >>> migo.gtp.parse_color('w') == migo.Color.WHITE
    True
    """
    if color_str.lower() == 'b':
        return Color.BLACK

    if color_str.lower() == 'w':
        return Color.WHITE

    raise ValueError("Cannot parse string `%s`" % color_str)


def parse_move(move_str: str) -> Move:
    """
    Parses a given string and returns Tuple of int as (row, col) or None
    as per the given string.
    If the string is not valid, ValueError is thrown.

    >>> migo.gtp.parse_move('PASS') is None
    True
    >>> migo.gtp.parse_move('A19')
    (18, 0)
    >>> migo.gtp.parse_move('J1')
    (0, 8)
    """
    if move_str.lower() == 'pass':
        return None

    matched = _MATCHER.match(move_str)
    if matched:
        alphabet = matched.group('alphabet').upper()
        number = matched.group('numeric')

        col = _LETTERS.find(alphabet)
        row = int(number) - 1

        if 0 <= row < 19 and 0 <= col < 19:
            return row, col

    matched = _MATCHER_cgoban.match(move_str)
    if matched:
        alphabet = matched.group('alphabet').upper()
        alphabet_col = matched.group('col').upper()

        row = _LETTERS_cgoban.find(alphabet)
        col = _LETTERS_cgoban.find(alphabet_col)

        if 0 <= row < 19 and 0 <= col < 19:
            return row, col

    raise ValueError("Cannot parse string `%s`" % move_str)


def move_to_str(move: Move) -> Optional[str]:
    """
    Returns a GTP string as per the given Move instance.
    e.g., "PASS", "A19", "J1" if the argument is None, (0, 18), (8, 0),
    respectively.

    >>> migo.gtp.move_to_str(None)
    'PASS'
    >>> migo.gtp.move_to_str((0, 18))
    'A19'
    >>> migo.gtp.move_to_str((8, 0))
    'J1'
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
