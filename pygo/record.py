import cygo
from .sgfutils import parse_one_game
import recordclass
import numpy as np


SimpleRecord = recordclass.recordclass(
    'SimpleRecord', (
        'board_size', 'komi', 'moves', 'winner', 'score',
     ))                         #: data record for a ordinary single game


def parse_sgf_game(sgf_string: str) -> SimpleRecord:
    '''read sgf_string containing a single game

    :return: :py:class:`SimpleRecord`

    >>> sgf_string = ('(;GM[1]SZ[9]KM[7.0]RE[B+26]RU[Chinese];B[de]'
    ...               'C[comment1];W[tt]C[comment2];B[tt]C[comment3])')
    >>> record = pygo.record.parse_sgf_game(sgf_string)
    >>> record.board_size
    9
    >>> record.komi
    7.0
    >>> record.winner == cygo.Color.BLACK
    True
    >>> state = cygo.State(record.board_size)
    >>> cygo.apply_moves(state, record.moves)
    >>> print(state)
      A  B  C  D  E  F  G  H  J 
    9 .  .  .  .  .  .  .  .  . 9
    8 .  .  .  .  .  .  .  .  . 8
    7 .  .  .  .  .  .  .  .  . 7
    6 .  .  .  .  .  .  .  .  . 6
    5 .  .  .  .  .  .  .  .  . 5
    4 .  .  .  .  X  .  .  .  . 4
    3 .  .  .  .  .  .  .  .  . 3
    2 .  .  .  .  .  .  .  .  . 2
    1 .  .  .  .  .  .  .  .  . 1
      A  B  C  D  E  F  G  H  J 
    ...
    '''
    state, moves, winner, score = parse_one_game(sgf_string, 0, cygo)
    board_size = state.board_size
    if state.current_player != cygo.Color.BLACK:
        raise RuntimeError('not implemented')
    if state.zobrist_hash != 0:
        raise RuntimeError('not implemented')

    raw_moves = np.zeros(len(moves), dtype=np.int32)
    for i, move in enumerate(moves):
        player_expected = cygo.Color.BLACK if i % 2 == 0 else cygo.Color.WHITE
        if move.player != player_expected:
            raise RuntimeError('not implemented')
        move = move.move
        if not move:
            raw_moves[i] = -1   # pass
            continue
        cmove = cygo.Move.from_coordinate(*move, board_size=board_size)
        raw_moves[i] = cmove.raw()

    return SimpleRecord(
        board_size=board_size,
        komi=state.komi,
        moves=raw_moves,
        winner=winner,
        score=score
    )


def load_sgf_game(sgf_path) -> SimpleRecord:
    '''open sgf file and return :py:class:`SimpleRecord`.
    '''
    with open(sgf_path) as file:
        content = file.read()
    return parse_sgf_game(content)


def load_sgf_games_in_folder(folder_path) -> list[SimpleRecord]:
    '''open sgf files in given folder to make list of :py:class:`SimpleRecord`
    '''
    import os.path
    import glob

    if not os.path.isdir(folder_path):
        raise RuntimeError(f'not folder {folder_path}')
    file_pattern = f'{folder_path}/*.sgf'
    games = []
    for sgf_path in glob.glob(file_pattern):
        games.append(load_sgf_game(sgf_path))
    return games
