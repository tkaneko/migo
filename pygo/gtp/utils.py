import re

from typing import Optional

from pygo import Color, Move

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
    :return: either :py:class:`pygo.Color.BLACK` or :py:class:`pygo.Color.WHITE`
    :exception ValueError: if `color_str` is an invalid gtp color string

    >>> pygo.gtp.parse_color('b') == pygo.Color.BLACK
    True
    >>> pygo.gtp.parse_color('w') == pygo.Color.WHITE
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

    >>> pygo.gtp.parse_move('PASS') is None
    True
    >>> pygo.gtp.parse_move('A19')
    (0, 18)
    >>> pygo.gtp.parse_move('J1')
    (8, 0)
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

    >>> pygo.gtp.move_to_str(None)
    'PASS'
    >>> pygo.gtp.move_to_str((0, 18))
    'A19'
    >>> pygo.gtp.move_to_str((8, 0))
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
