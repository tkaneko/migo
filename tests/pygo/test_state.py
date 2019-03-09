from random import Random

import numpy as np
import numpy.testing as npt

from pygo import State, Color, IllegalMoveError, all_coordinates


def parse(board_str: str, next_color: Color = None):
    board_str = board_str.replace(' ', '')
    size = max(board_str.index('|'), board_str.count('|'))

    state = State(board_size=size)

    moves = {}

    for row, row_str in enumerate(board_str.split('|')):
        for col, c in enumerate(row_str):
            if c == '.':
                continue

            if c in 'BX#':
                state.make_move((row, col), color=Color.BLACK)
            elif c in 'WO':
                state.make_move((row, col), color=Color.WHITE)
            else:
                moves[c] = (row, col)

    if next_color:
        state.current_player = next_color

    return state, moves


def random_state(seed=None):
    import os
    import sys

    seed = seed or int.from_bytes(os.urandom(4), sys.byteorder)

    random = Random(seed)

    size = random.randint(3, 19)
    length = random.randint(0, size * size)

    state = State(board_size=size)

    for i in range(length):
        move = (random.randrange(0, size), random.randrange(0, size))

        try:
            state.make_move(move)
        except IllegalMoveError:
            pass

    return state, seed


def test_parse_board():
    state, moves = parse(
        "1 2 3 . . . . |"
        "B W . . . . . |",
        next_color=Color.BLACK
    )

    assert state.current_player == Color.BLACK
    assert state.board_size == 7

    assert moves['1'] == (0, 0)
    assert moves['2'] == (0, 1)
    assert moves['3'] == (0, 2)

    # labels should be empty
    assert state.color_at(moves['1']) == Color.EMPTY
    assert state.color_at(moves['2']) == Color.EMPTY
    assert state.color_at(moves['3']) == Color.EMPTY

    assert state.color_at((1, 0)) == Color.BLACK
    assert state.color_at((1, 1)) == Color.WHITE
    assert state.color_at((1, 2)) == Color.EMPTY


def test_copy():
    state, _ = parse(
        ". B . . . . . . .|"
        "B B W . . . . . .|"
        ". B W W . . . . .|"
        "B B W . . . . . .|"
        ". . W W . . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
    )

    copied = state.copy()

    assert state is not copied

    assert state.groups is not copied.groups
    assert state.groups == copied.groups

    assert state.liberty_sets is not copied.liberty_sets
    assert state.liberty_sets == copied.liberty_sets

    assert state.liberty_counts is not copied.liberty_counts
    npt.assert_array_equal(state.liberty_counts, copied.liberty_counts)

    assert state.board is not copied.board
    npt.assert_array_equal(state.board, copied.board)

    assert state.zobrist_hash == copied.zobrist_hash


def test_groups_emtpy():
    state, seed = random_state()

    msg = 'random_state(seed=%s)' % seed

    for position in state.empties:
        assert state.stone_age_of(position) == -1, msg
        assert state.liberty_count_of(position) == -1, msg
        assert len(state.group_at(position)) == 0, msg


def test_when_merge_happens_then_counts_correctly():
    state, moves = parse(
        "B a .|"
        ". B .|"
        ". . .|",
        next_color=Color.BLACK
    )

    assert len(state.group_at((0, 0))) == 1
    assert len(state.group_at((1, 1))) == 1

    state.make_move(moves['a'])

    assert len(state.group_at((0, 0))) == 3
    assert state.group_at((0, 0)) == {(0, 0), (0, 1), (1, 1)}

    assert state.groups[(0, 0)] is state.groups[(0, 1)]
    assert state.groups[(0, 0)] is state.groups[(1, 1)]


def test_get_emtpy_coord_when_there_are_stones():
    state, moves = parse(
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . 1 . . . .|"
        ". . . . . . . . .|"
        ". . . . . . . . .|"
        ". . . . . 2 . . .|"
        ". . . . . . . . .|"
        ". . . . . . 3 . .|"
    )

    for move in moves.values():
        state.make_move(move)

    expected_set = set(all_coordinates(state.board_size)) - set(moves.values())
    actual_set = state.empties

    assert expected_set == actual_set


def test_liberties_after_capture():
    state, moves = parse(
        ". . . . . . .|"
        ". . W W W . .|"
        ". W B B B a .|"
        ". W B B B W .|"
        ". W B B B W .|"
        ". . W W W . .|"
        ". . . . . . .|"
    )

    expected_state, _ = parse(
        ". . . . . . .|"
        ". . W W W . .|"
        ". W . . . W .|"
        ". W . . . W .|"
        ". W . . . W .|"
        ". . W W W . .|"
        ". . . . . . .|"
    )

    state.make_move(moves['a'], Color.WHITE)

    npt.assert_array_equal(expected_state.board, state.board)
    npt.assert_array_equal(expected_state.liberty_counts, state.liberty_counts)

    assert expected_state.liberty_sets == state.liberty_sets
    assert expected_state.groups == state.groups


def test_copy_maintains_shared_sets():
    state = State(7)
    state.make_move((4, 4), Color.BLACK)
    state.make_move((4, 5), Color.BLACK)

    # assert that state has *the same object* referenced by group/liberty sets
    assert state.groups[(4, 5)] is state.groups[(4, 4)]
    assert state.liberty_sets[(4, 5)] is state.liberty_sets[(4, 4)]

    copied = state.copy()
    assert copied.groups[(4, 5)] is copied.groups[(4, 4)]
    assert copied.liberty_sets[(4, 5)] is copied.liberty_sets[(4, 4)]


class TestSuicideMove:
    def test_1(self):
        state, moves = parse(
            ". B .|"
            "B a B|"
            ". B .|")

        assert state.is_suicide_move(moves['a'], Color.WHITE)
        assert not state.is_suicide_move(moves['a'], Color.BLACK)

    def test_2(self):
        state, moves = parse(
            ". . . . B a|"
            ". . . . . B|"
            ". . . . . .|"
            ". . . . . .|"
            ". . . . . .|"
            ". . . . . .|")

        assert state.is_suicide_move(moves['a'], Color.WHITE)
        assert not state.is_suicide_move(moves['a'], Color.BLACK)


class TestKo:
    def test_when_ko_occurred_then_recognizes_as_ko(self):
        state, moves = parse(
            ". W B .|"
            "W 1 W B|"
            ". W B .|",
            next_color=Color.BLACK
        )

        expected_ko = (1, 2)

        state.make_move(moves['1'])

        assert expected_ko == state.ko

    def test_when_recapturable_then_it_is_not_ko(self):
        state, moves = parse(
            "B 1 W B|"
            "W W B .|",
            next_color=Color.BLACK
        )

        state.make_move(moves['1'])

        assert state.ko is None
        assert state.is_legal((2, 0), Color.WHITE)

    def test_when_in_superko_rule(self):
        # http://senseis.xmp.net/?PositionalSuperko%2FExample
        state, moves = parse(
            ". . . . . . . . .|"
            ". . B . . . . . .|"
            ". . B . . . . . .|"
            ". . B W W W W . .|"
            ". B 2 B 1 W . W .|"
            ". . B W W W W . .|"
            ". . B . . . . . .|"
            ". . B . . . . . .|"
            ". . . . . . . . .|",
            next_color=Color.BLACK)

        expected_positional_superko_move = (4, 3)

        state.make_move(moves['1'])
        state.make_move(moves['2'])

        assert state.is_positional_superko(expected_positional_superko_move)


class TestEye:
    def test_eyeish(self):
        state, moves = parse(
            ". B . B .|"
            "B a B b B|"
            ". B . B .|"
        )

        assert state.is_eyeish(moves['a'], Color.BLACK)
        assert state.is_eyeish(moves['b'], Color.BLACK)

    def test_eye(self):
        state, moves = parse(
            "c X X X X|"
            "X a X b X|"
            "X X X X d|"
            ". . . . .|"
            ". . . . .|"
        )

        assert state.is_eye(moves['a'], Color.BLACK)
        assert state.is_eye(moves['b'], Color.BLACK)
        assert state.is_eye(moves['c'], Color.BLACK)
        assert not state.is_eye(moves['d'], Color.BLACK)

        assert state.is_eyeish(moves['a'], Color.BLACK)
        assert state.is_eyeish(moves['b'], Color.BLACK)
        assert state.is_eyeish(moves['c'], Color.BLACK)
        assert not state.is_eyeish(moves['d'], Color.BLACK)


class TestLiberties:
    def test_liberty_1(self):
        state, moves = parse(
            ". . . . . . . . .|" 
            ". B W . . . . . .|" 
            ". B B B . . . . .|" 
            ". . . . . . . . .|" 
            ". . . . . . . . .|" 
            ". . . . . . . . .|" 
            ". . . . . . W W .|" 
            ". . . . . . . W .|" 
            ". . . . . . . . .|"
        )

        assert state.liberty_count_of((1, 2)) == 2
        assert state.liberty_count_of((1, 1)) == 8
        assert state.liberty_count_of((2, 1)) == 8

        assert state.liberty_count_of((6, 6)) == 7
        assert state.liberty_count_of((6, 7)) == 7
        assert state.liberty_count_of((7, 7)) == 7

    def test_liberty_2(self):
        state, _ = parse(
            "B B . . .|"
            "B W . . .|"
            ". . . . .|"
            ". . . W .|"
            ". . . . W|"
        )

        assert state.liberty_count_of((0, 0)) == 2
        assert state.liberty_count_of((1, 0)) == 2
        assert state.liberty_count_of((0, 1)) == 2
        assert state.liberty_count_of((4, 4)) == 2
        assert state.liberty_count_of((3, 3)) == 4


class TestLadder:
    def test_captured_1(self):
        state, moves = parse(
            "d b c . . . .|"
            "B W a . . . .|"
            ". B . . . . .|"
            ". . . . . . .|"
            ". . . . . . .|"
            ". . . . . W .|",
            next_color=Color.BLACK)

        assert state.is_ladder_capture(moves['a'])
        assert not state.is_ladder_capture(moves['b'])

        state.make_move(moves['a'])

        assert not state.is_ladder_escape(moves['b'])

        state.make_move(moves['b'])

        assert state.is_ladder_capture(moves['c'])
        assert not state.is_ladder_capture(moves['d'])  # self-atari

    def test_breaker_1(self):
        state, moves = parse(
            ". B . . . . .|"
            "B W a . . W .|"
            "B b . . . . .|"
            ". c . . . . .|"
            ". . . . . . .|"
            ". . . . . W .|"
            ". . . . . . .|",
            next_color=Color.BLACK
        )

        # 'a' should not be a ladder capture, nor 'b'
        assert not state.is_ladder_capture(moves['a'])
        assert not state.is_ladder_capture(moves['b'])

        # after 'a', 'b' should be an escape
        state.make_move(moves['a'])
        assert state.is_ladder_escape(moves['b'])

        # after 'b', 'c' should not be a capture
        state.make_move(moves['b'])
        assert not state.is_ladder_capture(moves['c'])

    def test_missing_ladder_breaker_1(self):
        state, moves = parse(
            ". B . . . . .|"
            "B W B . . W .|"
            "B a c . . . .|"
            ". b . . . . .|"
            ". . . . . . .|"
            ". W . . . . .|"
            ". . . . . . .|",
            next_color=Color.WHITE
        )

        # a should not be an escape move for white
        assert not state.is_ladder_escape(moves['a'])

        # after 'a', 'b' should still be a capture ...
        state.make_move(moves['a'])
        assert state.is_ladder_capture(moves['b'])
        # ... but 'c' should not
        assert not state.is_ladder_capture(moves['c'])

    def test_capture_to_escape_1(self):
        state, moves = parse(
            ". W B * . .|"
            ". B W B . .|"
            ". . W B . .|"
            ". . a . . .|"
            ". B . . . .|"
            ". . . . . .|",
            next_color=Color.BLACK
        )

        # 'a' is not a capture, since white can capture black by playing '*'
        assert not state.is_ladder_capture(moves['a'])

    def test_throw_in_1(self):
        state, moves = parse(
            "B a W B . .|"
            "b W W B . .|"
            "W W B B . .|"
            "B B . . . .|"
            ". . . . . .|"
            ". . . W . .|",
            next_color=Color.BLACK
        )

        # 'a' or 'b' will capture
        assert state.is_ladder_capture(moves['a'])
        assert state.is_ladder_capture(moves['b'])

        # after 'a', 'b' doesn't help white escape
        state.make_move(moves['a'])
        assert not state.is_ladder_escape(moves['b'])

    def test_snapback_1(self):
        state, moves = parse(
            ". . . . . . . . .|"
            ". . . . . . . . .|"
            ". . B B B . . . .|"
            ". . W . . . . . .|"
            ". . W B . . . . .|"
            ". . B W a . . . .|"
            ". . B W B . . . .|"
            ". . . B . . . . .|"
            ". . . . . . . . .|",
            next_color=Color.WHITE
        )

        # 'a' is not an escape for white
        assert not state.is_ladder_escape(moves['a'])

    def test_two_captures(self):
        state, moves = parse(
            ". . . . . .|"
            ". . . . . .|"
            ". . a b . .|"
            ". B W W B .|"
            ". . B B . .|"
            ". . . . . .|",
            next_color=Color.BLACK
        )

        # both 'a' and 'b' should be ladder captures
        assert state.is_ladder_capture(moves['a'])
        assert state.is_ladder_capture(moves['b'])

    def test_two_escapes(self):
        state, moves = parse(
            ". . B . . .|"
            ". B W a . .|"
            ". B c B . .|"
            ". W B b . .|"
            ". . W . . .|"
            ". . . . . .|",
            next_color=Color.WHITE
        )

        # place a white stone at c, and reset player to white
        state.make_move(moves['c'])
        state.current_player = Color.WHITE

        # both 'a' and 'b' should be considered escape moves for white after 'O' at c
        assert state.is_ladder_escape(moves['a'])
        assert state.is_ladder_escape(moves['b'], prey=moves['c'])


class TestHistory:
    def test_history(self):
        size = 3
        state = State(board_size=size, history_buffer_len=10)

        expected_boards = [
            np.array([[+1, +0, +0],
                      [+0, +0, +0],
                      [+0, +0, +0]]),
            np.array([[+1, -1, +0],
                      [+0, +0, +0],
                      [+0, +0, +0]]),
            np.array([[+1, -1, +1],
                      [+0, +0, +0],
                      [+0, +0, +0]])
        ]

        state.make_move((0, 0))
        state.make_move((0, 1))
        state.make_move((0, 2))

        assert len(state.history_buffer) == len(expected_boards)

        npt.assert_equal(state.history_buffer[0], expected_boards[0])
        npt.assert_equal(state.history_buffer[1], expected_boards[1])
        npt.assert_equal(state.history_buffer[2], expected_boards[2])

    def test_history_pop(self):
        size = 3
        state = State(board_size=size, history_buffer_len=3)

        expected_boards = [
            np.array([[+1, +0, +0],
                      [+0, +0, +0],
                      [+0, +0, +0]]),
            np.array([[+1, -1, +0],
                      [+0, +0, +0],
                      [+0, +0, +0]]),
            np.array([[+1, -1, +1],
                      [+0, +0, +0],
                      [+0, +0, +0]])
        ]

        state.make_move((0, 0))
        state.make_move((0, 1))
        state.make_move((0, 2))

        assert len(state.history_buffer) == 3

        npt.assert_equal(state.history_buffer[0], expected_boards[0])
        npt.assert_equal(state.history_buffer[1], expected_boards[1])
        npt.assert_equal(state.history_buffer[2], expected_boards[2])

    def test_history_no_history(self):
        size = 3
        state = State(board_size=size, history_buffer_len=0)

        assert len(state.history_buffer) == 0

        state.make_move((0, 0))

        assert len(state.history_buffer) == 0
