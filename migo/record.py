import cygo
from .sgfutils import parse_one_game, SgfPrinter
import recordclass
import numpy as np
import io


SimpleRecord = recordclass.recordclass(
    'SimpleRecord', (
        'board_size', 'komi',
        # results
        'moves', 'winner', 'score',
        'territory', 'zone_score',
    ))                         #: data record for a ordinary single game


def parse_sgf_game(sgf_string: str) -> SimpleRecord:
    '''read sgf_string containing a single game

    :return: :py:class:`SimpleRecord`

    >>> sgf_string = ('(;GM[1]SZ[9]KM[7.0]RE[B+74]RU[Chinese];B[de]'
    ...               'C[comment1];W[tt]C[comment2];B[tt]C[comment3])')
    >>> record = migo.record.parse_sgf_game(sgf_string)
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
    5 .  .  .  X  .  .  .  .  . 5
    4 .  .  .  .  .  .  .  .  . 4
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

    raw_moves = np.zeros(len(moves), dtype=np.int16)
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


def move_to_sgf(move):
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    if isinstance(move, cygo.Move):
        r, c = move.row, move.col
    if isinstance(move, tuple):
        r, c = move
    return ascii_lowercase[c] + ascii_lowercase[r]


def record_to_sgf(record: SimpleRecord) -> str:
    '''return sgf representation of given record'''
    out = io.StringIO()
    props = {'KM': f'{record.komi}'}
    if record.score is not None:
        if record.score == 0:
            props['RE'] = 'draw'
        elif record.score > 0:
            props['RE'] = f'B+{record.score}'
        else:
            props['RE'] = f'W+{-record.score}'
    prt = SgfPrinter(
        out, size=record.board_size,
        initial_props=props
    )
    prt.open()
    color = 'BW'
    for i, move in enumerate(record.moves):
        label = ''
        if move >= 0:
            move = cygo.Move.from_raw_value(move, record.board_size)
            label = move_to_sgf(move)
        prt.add_node({color[i % 2]: label})
    prt.close()
    return out.getvalue()
