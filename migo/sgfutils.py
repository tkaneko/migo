"""sgf related functions

As `sgf` package has sufficient functionalities for input (e.g.,
`sgf.parse(string)`) and outputs (`collection.output(file)`), this
file provides supplementary staff specialized for go and simple game
records without any branch.

- base package https://github.com/jtauber/sgf
- specifications https://red-bean.com/sgf/
"""
import sgf
import recordclass
import re
import logging
from enum import Enum
from string import ascii_lowercase, ascii_uppercase
from typing import Optional, Tuple, Dict, List, Union, Generator


_SGF_SCORE_REGEX = re.compile(r'(((?P<color>[BW])\+(?P<score>(\d+|R)))|DRAW)',
                              re.IGNORECASE)
_SGF_MOVE_REGEX = re.compile(r'^[A-T]{2}$', re.IGNORECASE)


class SgfColor(Enum):
    BLACK = 'B'
    WHITE = 'W'
    NIL = ''


class SizeMismatchError(Exception):
    pass


class SgfContentError(Exception):
    pass


def move_to_sgf(move: Optional[Tuple[int, int]]) -> str:
    """return sgf representation of Coord

    >>> migo.sgfutils.move_to_sgf((0, 0))
    'aa'
    >>> migo.sgfutils.move_to_sgf((1, 2))
    'bc'
    >>> migo.sgfutils.move_to_sgf(None)
    'tt'
    """
    if move is None:
        return 'tt'

    return ascii_lowercase[move[0]] + ascii_lowercase[move[1]]


def parse_sgf_move(
        move_str: str,
        board_size: int = 0
) -> Optional[Tuple[int, int]]:
    """Returns either None or (x, y) coordinates of board.

    :param move_str: two letters like 'ab'
    :param board_size: board size, 0 is allowed only for pass
    """
    if not (move_str == '' or _SGF_MOVE_REGEX.match(move_str)):
        raise SgfContentError

    if move_str == '' or move_str == 'tt':
        return None

    if board_size < 1:
        raise SgfContentError("need board_size if a move is not pass")

    e = SgfContentError("Invalid move string %s" % move_str)

    if len(move_str) != 2:
        raise e

    try:
        # GameState expects (x, y) where x is column and y is row
        col = ascii_uppercase.index(move_str[0].upper())
        row = ascii_uppercase.index(move_str[1].upper())

        # flip vertically to be consistent with gogui
        row = board_size - 1 - row

        if (0 <= col < 19) and (0 <= row < 19):
            return row, col

        raise e

    except ValueError:
        raise e


def parse_sgf_result(score_str: str) -> Tuple[SgfColor, float]:
    if not _SGF_SCORE_REGEX.match(score_str):
        raise SgfContentError("Cannot parse score string `%s`" % score_str)

    if score_str.lower() == 'draw':
        return SgfColor.NIL, 0.0

    if score_str[0].lower() == 'b':
        winner = SgfColor.BLACK
    elif score_str[0].lower() == 'w':
        winner = SgfColor.WHITE
    else:
        raise SgfContentError

    if score_str[2] == 'R':
        score = 0.
    else:
        score = float(score_str[2:])

    return winner, score


def game_length(sgf_string: str) -> int:
    collection = sgf.parse(sgf_string)
    if len(collection) != 1:
        logging.warning(f'sgf_string contains {len(collection)} records')

    game = collection[0]

    if not game.rest:
        return 0

    num_moves = 0
    for node in game.rest:
        if 'W' in node.properties or 'B' in node.properties:
            num_moves += 1

    return num_moves


class SgfPrinter:
    """incrementally output sgf by `add_node`

    >>> out = io.StringIO()
    >>> prt = migo.sgfutils.SgfPrinter(out, size=4)
    >>> _ = prt.open()
    >>> prt.add_node({'B': 'a2'})
    >>> prt.add_node({'W': '2c'})
    >>> prt.close()
    >>> print(out.getvalue())
    (;FF[4]GM[1]SZ[4];B[a2];W[2c])
    """
    def __init__(self, out, size, initial_props: Optional[Dict] = None):
        self.parser = sgf.Parser()
        self.collection = sgf.Collection(self.parser)

        self.size = size
        self.out = out

        self.initial_props = initial_props or {}

    def __enter__(self):
        return self.open()

    def open(self):
        """start and open a game tree"""
        self.parser.start_gametree()
        self.__add_initial_props()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """close the game tree previously opened by open()"""
        self.parser.end_gametree()

        self.collection.output(self.out)

    def add_node(self, dic: Dict[str, Union[str, List[str]]]):
        """add a node in the game tree"""
        self.parser.start_node()

        for key, value in dic.items():
            self.__add_property(key, value)

        self.parser.end_node()

    def __add_initial_props(self):
        self.parser.start_node()

        self.__add_property('FF', '4')
        self.__add_property('GM', '1')
        self.__add_property('SZ', str(self.size))

        for key, value in self.initial_props.items():
            self.__add_property(key, value)

    def __add_property(self, identifier, value):
        self.parser.start_property(identifier)

        if isinstance(value, list):
            for v in value:
                self.parser.add_prop_value(v)
        else:
            self.parser.add_prop_value(value)

        self.parser.end_property()


def _set_up_state(props: Dict[str, List[str]], board_size: int, go):
    try:
        size = int(props['SZ'][0])
        if board_size > 0:
            assert size == board_size, "Expected in {e}x{e}" \
                " but game was in {a}x{a}".format(a=size, e=board_size)
        board_size = size
        komi = float(props['KM'][0])

    except (KeyError, AssertionError) as e:
        raise SgfContentError(e)

    start_player = props.get('PL', ['B'])[0]

    state = go.State(size, komi=komi)

    for sgf_move in props.get('AB', []):
        move = parse_sgf_move(sgf_move, board_size)
        state.make_move(move, go.Color.BLACK)

    for sgf_move in props.get('AW', []):
        move = parse_sgf_move(sgf_move, board_size)
        state.make_move(move, go.Color.WHITE)

    state.current_player = go.Color.BLACK if start_player == 'B' \
        else go.Color.WHITE

    state.drop_history()
    return state


MoveRecord = recordclass.recordclass(
    'MoveRecord', (
        'move', 'player', 'comment'
     ))


def parse_one_game(sgf_string: str, board_size: int, go,
                   *, allow_ongoing_game: bool = False):
    """parse sgf to return initial state, moves, winner, score

    :param board_size: expected board size or 0 (for any)
    :param go: :py:mod:`migo` or :py:mod:`cygo`
    """
    collection = sgf.parse(sgf_string)
    game = collection[0]

    root_props = game.root.properties

    initial_state = _set_up_state(root_props, board_size, go)
    board_size = initial_state.board_size

    moves = []
    for node in game.rest or []:
        props = node.properties

        if 'W' in props:
            move = parse_sgf_move(props['W'][0], board_size)
            player = go.Color.WHITE
        elif 'B' in props:
            move = parse_sgf_move(props['B'][0], board_size)
            player = go.Color.BLACK
        else:
            raise SgfContentError("Found a node without move properties")

        comment = props.get('C', [''])[0]
        moves.append(MoveRecord(move, player, comment))

    if 'RE' not in root_props:
        if allow_ongoing_game:
            return initial_state, moves, go.Color.EMPTY, 0
        raise SgfContentError("SGF file does not contain 'RE' property")

    winner, score = parse_sgf_result(root_props['RE'][0])

    if winner == SgfColor.BLACK:
        winner = go.Color.BLACK
    elif winner == SgfColor.WHITE:
        winner = go.Color.WHITE
    else:
        winner = go.Color.EMPTY

    return initial_state, moves, winner, score


def sgf_generator(sgf_string: str, board_size: int, go,
                  include_end=True) -> Generator[Tuple, None, None]:
    """make a generator yielding state, move, winner, score, comment

    :param go: :py:mod:`migo` or :py:mod:`cygo`
    """
    state, moves, winner, score = parse_one_game(sgf_string, board_size, go)
    for move in moves:
        state.current_player = move.player

        yield state, move.move, winner, score, move.comment
        state.make_move(move.move)

    if include_end:
        yield state, None, winner, score, ''
