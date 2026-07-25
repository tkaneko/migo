from __future__ import annotations
import numpy
import typing
from . import features
__all__: list[str] = ['BLACK', 'Color', 'EMPTY', 'Move', 'Pass', 'State', 'WHITE', 'apply_moves', 'features', 'opposite_color', 'zobrist_hash']
class Color:
    """
    define color id following :cpp:enum:`cygo::Color`
    
    Members:
    
      BLACK
    
      WHITE
    
      EMPTY
    """
    BLACK: typing.ClassVar[Color]  # value = <Color.BLACK: 1>
    EMPTY: typing.ClassVar[Color]  # value = <Color.EMPTY: 0>
    WHITE: typing.ClassVar[Color]  # value = <Color.WHITE: -1>
    __members__: typing.ClassVar[dict[str, Color]]  # value = {'BLACK': <Color.BLACK: 1>, 'WHITE': <Color.WHITE: -1>, 'EMPTY': <Color.EMPTY: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def opponent(self) -> Color:
        """
        Return the opposite color
        
        >>> cygo.BLACK.opponent() == cygo.WHITE
        True
        """
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Move:
    Pass: typing.ClassVar[Move]  # value = Move.Pass
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_coordinate(row: int, col: int, board_size: int) -> Move:
        """
        construct Move object from coordinate and board size
        
        >>> move = cygo.Move.from_coordinate(1, 2, board_size=4)
        >>> move
        (1, 2)
        >>> move.row
        1
        >>> move.col
        2
        >>> move.board_size
        4
        >>> move.raw()
        6
        >>> move.n
        (1, 1)
        >>> move.s
        (1, 3)
        >>> move.w
        (0, 2)
        >>> move.e
        (2, 2)
        """
    @staticmethod
    def from_gtp_string(gtp_string: str, board_size: int) -> Move:
        """
        construct Move object from gtp string and board size
        
        >>> a1 = cygo.Move.from_gtp_string('a1', board_size=4)
        >>> a1
        (0, 0)
        >>> a1.is_at_corner
        True
        >>> a1.is_on_edge
        True
        >>> c2 = cygo.Move.from_gtp_string('c2', 4)
        >>> c2
        (1, 2)
        >>> c2.is_on_edge
        False
        >>> c2.gtp
        'C2'
        >>> pass_move = cygo.Move.from_gtp_string('pass', 4)
        >>> pass_move
        Move.Pass
        >>> pass_move.gtp
        'PASS'
        """
    @staticmethod
    def from_raw_value(raw_value: int, board_size: int) -> Move:
        """
        construct Move object from internal representation
        
        >>> move = cygo.Move.from_raw_value(6, board_size=4)
        >>> move
        (1, 2)
        """
    def __eq__(self, arg0: Move) -> bool:
        ...
    def __ne__(self, arg0: Move) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def raw(self) -> int:
        ...
    @property
    def board_size(self) -> int:
        """
        board_size in int
        """
    @property
    def col(self) -> int:
        """
        col in int
        """
    @property
    def e(self) -> Move:
        """
        move at neighbor
        """
    @property
    def gtp(self) -> str:
        """
        gtp representation
        """
    @property
    def is_at_corner(self) -> bool:
        """
        true if on any of four corners
        """
    @property
    def is_on_edge(self) -> bool:
        """
        true if on edge
        """
    @property
    def is_pass(self) -> bool:
        """
        true if pass
        """
    @property
    def n(self) -> Move:
        """
        move at neighbor
        """
    @property
    def row(self) -> int:
        """
        row in int
        """
    @property
    def s(self) -> Move:
        """
        move at neighbor
        """
    @property
    def w(self) -> Move:
        """
        move at neighbor
        """
class State:
    """
    Go state
    
    :param board_size: int
    :param komi: float = 7.5
    :param superko_rule: bool = True
    :param max_history_n: int = 7
    
    >>> state = cygo.State(4)
    >>> print(state)
      A  B  C  D
    4 .  .  .  . 4
    3 .  .  .  . 3
    2 .  .  .  . 2
    1 .  .  .  . 1
      A  B  C  D
    ...
    >>> state.make_move((2, 1))
    >>> print(state)
      A  B  C  D
    4 .  .  .  . 4
    3 . (X) .  . 3
    2 .  .  .  . 2
    1 .  .  .  . 1
      A  B  C  D
    ...
    >>> state.make_move('c3')
    >>> print(state)
      A  B  C  D
    4 .  .  .  . 4
    3 .  X (O) . 3
    2 .  .  .  . 2
    1 .  .  .  . 1
      A  B  C  D
    ...
    >>> state.last_move
    (2, 2)
    """
    def __init__(self, board_size: int, komi: float = 7.5, superko_rule: bool = True, max_history_n: int = 7) -> None:
        ...
    def __str__(self) -> str:
        ...
    def color_at(self, vertex: tuple[int, int]) -> Color:
        """
        return cygo::Color at vertex
        """
    def color_move_history(self, color: Color) -> list[Move]:
        """
        Returns color's move history list
        """
    def copy(self) -> State:
        ...
    def drop_history(self) -> None:
        """
        drop history keeping current stones
        """
    def info(self) -> str:
        """
        return internal info
        """
    @typing.overload
    def is_eye_like(self, move: Move, color: Color = ...) -> bool:
        ...
    @typing.overload
    def is_eye_like(self, move: tuple[int, int], color: Color = ...) -> bool:
        ...
    @typing.overload
    def is_legal(self, move: Move | None, color: Color = ...) -> bool:
        ...
    @typing.overload
    def is_legal(self, move: str, color: Color = ...) -> bool:
        ...
    @typing.overload
    def is_legal(self, move: tuple[int, int], color: Color = ...) -> bool:
        """
        Return move is legal
        """
    def is_suicide_move(self, move: Move, color: Color = ...) -> bool:
        """
        Return whether move is suicide
        """
    def legal_indices(self, color: Color = ..., include_eyeish: bool = True) -> list[int]:
        """
        Generate legal indices for the current state
        """
    def legal_moves(self, color: Color = ..., include_eyeish: bool = True) -> set[tuple[int, int]]:
        """
        Generate legal moves for the current state
        """
    @typing.overload
    def make_move(self, index: int, color: Color = ...) -> None:
        """
        Apply move to the state as color
        
        :param index: index acceptable by :cpp:func:`cygo::Move::from_raw` or -1 for pass
        """
    @typing.overload
    def make_move(self, move: Move | None, color: Color = ...) -> None:
        """
        Apply move to the state as color
        
        :param move: None for pass
        """
    @typing.overload
    def make_move(self, move: str, color: Color = ...) -> None:
        """
        :param move: string acceptable by :py:meth:`cygo.Move.from_gtp_string`
        """
    @typing.overload
    def make_move(self, move: tuple[int, int], color: Color = ...) -> None:
        """
        Apply move to the state as color
        """
    def tromp_taylor_fill(self) -> numpy.ndarray[numpy.int8]:
        """
        Return numpy array filled by color
        """
    def tromp_taylor_score(self, color: Color = ...) -> float:
        """
        >>> state = cygo.State(5, komi=1)
        >>> moves = ['a2', 'a3', 'b2', 'b3', 'c2', 'c3', 'd2', 'd3', 'e2', 'e3']
        >>> for m in moves:
        ...   state.make_move(m)
        >>> print(state)
          A  B  C  D  E
        5 .  .  .  .  . 5
        4 .  .  .  .  . 4
        3 O  O  O  O (O)3
        2 X  X  X  X  X 2
        1 .  .  .  .  . 1
          A  B  C  D  E
        <BLANKLINE>
        >>> state.tromp_taylor_score(cygo.Color.BLACK)
        -6.0
        >>> state.tromp_taylor_score()
        -6.0
        >>> state.make_move('C1')
        >>> print(state)
          A  B  C  D  E
        5 .  .  .  .  . 5
        4 .  .  .  .  . 4
        3 O  O  O  O  O 3
        2 X  X  X  X  X 2
        1 .  . (X) .  . 1
          A  B  C  D  E
        <BLANKLINE>
        >>> state.tromp_taylor_score(cygo.Color.BLACK)
        -6.0
        >>> state.tromp_taylor_score()
        6.0
        >>> state.make_move('D1')
        >>> print(state)
          A  B  C  D  E
        5 .  .  .  .  . 5
        4 .  .  .  .  . 4
        3 O  O  O  O  O 3
        2 X  X  X  X  X 2
        1 .  .  X (O) . 1
          A  B  C  D  E
        <BLANKLINE>
        >>> state.tromp_taylor_score(cygo.Color.BLACK)
        -9.0
        >>> state.make_move('E1')
        >>> print(state)
          A  B  C  D  E
        5 .  .  .  .  . 5
        4 .  .  .  .  . 4
        3 O  O  O  O  O 3
        2 X  X  X  X  X 2
        1 .  .  X  . (X)1
          A  B  C  D  E
        <BLANKLINE>
        >>> state.tromp_taylor_score(cygo.Color.BLACK)
        -6.0
        """
    @property
    def board_size(self) -> int:
        """
        Return the current board size
        
        :type: int
        """
    @property
    def current_player(self) -> Color:
        """
        Opposite of the last played color
        
        :type: :py:class:`cygo.Color`
        """
    @current_player.setter
    def current_player(self, arg1: Color) -> None:
        ...
    @property
    def komi(self) -> float:
        """
        Komi value
        
        :type: float
        """
    @komi.setter
    def komi(self, arg1: float) -> None:
        ...
    @property
    def last_move(self) -> Move:
        """
        Return the last move
        
        :type: :py:class:`cygo.Move`
        """
    @property
    def max_history_n(self) -> int:
        """
        maximum length of history to remember
        
        :type: int
        """
    @property
    def move_history(self) -> list[Move]:
        """
        Returns color's move history list
        """
    @property
    def superko_rule(self) -> bool:
        """
        True if superko is adopted
        
        :type: bool
        """
    @property
    def zobrist_hash(self) -> int:
        """
        return 64bit hash for board status (ignoring history) to quickly detect equivalence
        
        >>> state = cygo.State(5)
        >>> moves = ['B3', 'C2', 'C4']
        >>> print(state.zobrist_hash)
        0
        >>> for move in moves:
        ...   state.make_move(move)
        ...   print(state.zobrist_hash)
        12092317580524320504
        13032588549984992753
        493738324825164472
        >>> state2 = cygo.State(5)
        >>> for move in reversed(moves):
        ...   state2.make_move(move)
        >>> state.zobrist_hash == state2.zobrist_hash
        True
        """
def apply_moves(state: State, moves: numpy.ndarray[numpy.int32], move_id: int = -1) -> None:
    """
    Apply given moves to the given state.
    
    The moves should be an ndarray of ints, each of which is Move.raw or -1 for pass.
    
    .. warning:: -1 is inconsistent with :cpp:member:`cygo::Move::PASS` which is -2
    """
def opposite_color(color: Color) -> Color:
    """
    Return the opposite color of a given color
    """
def zobrist_hash(black_array: numpy.ndarray[numpy.float32], white_array: numpy.ndarray[numpy.float32]) -> int:
    """
    Calculate zobrist hash for a given position represented by two ndarrays
    """
BLACK: Color  # value = <Color.BLACK: 1>
EMPTY: Color  # value = <Color.EMPTY: 0>
Pass = None
WHITE: Color  # value = <Color.WHITE: -1>
