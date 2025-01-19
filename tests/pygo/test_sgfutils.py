import pytest

import cygo
import migo

from migo.sgfutils import SgfColor, SgfContentError, parse_sgf_move, \
    parse_sgf_result, sgf_generator, parse_one_game


class TestSgfParseSgfResult:
    def test_when_result_is_draw(self):
        winner, score = parse_sgf_result('Draw')

        assert winner == SgfColor.NIL
        assert score == 0.0

    def test_when_result_is_black_won_by_resignation(self):
        winner, score = parse_sgf_result('B+R')

        assert winner == SgfColor.BLACK
        assert score == 0.0

    def test_when_result_is_white_won_by_resignation(self):
        winner, score = parse_sgf_result('W+R')

        assert winner == SgfColor.WHITE
        assert score == 0.0

    def test_when_result_is_black_won(self):
        winner, score = parse_sgf_result('B+42.0')

        assert winner == SgfColor.BLACK
        assert score == 42.0

    def test_when_unrecognizable(self):
        with pytest.raises(SgfContentError):
            parse_sgf_result("")

        with pytest.raises(SgfContentError):
            parse_sgf_result("foobar")

        with pytest.raises(SgfContentError):
            parse_sgf_result("B+a")

        with pytest.raises(SgfContentError):
            parse_sgf_result("B32")


class TestSgfParseSgfMove:
    def test_when_move_is_pass(self):
        assert parse_sgf_move('') is None
        assert parse_sgf_move('tt') is None

    def test_when_move_is_valid(self):
        assert parse_sgf_move('aa') == (0, 0)
        assert parse_sgf_move('AA') == (0, 0)
        assert parse_sgf_move('ab') == (0, 1)
        assert parse_sgf_move('ss') == (18, 18)

    def test_when_move_is_invalid(self):
        with pytest.raises(SgfContentError):
            parse_sgf_move('foobar')

        with pytest.raises(SgfContentError):
            parse_sgf_move('st')

        with pytest.raises(SgfContentError):
            parse_sgf_move('aaa')


class TestSgfGenerator:
    @pytest.fixture(scope='class', autouse=True)
    def sgf_string(self):
        yield """
(;GM[1]SZ[9]KM[7.0]RE[B+26]RU[Chinese]
;B[de]C[comment1]
;W[tt]C[comment2]
;B[tt]C[comment3]
)
"""

    @pytest.mark.parametrize('go', [migo, cygo])
    def test_with_gomodule(self, sgf_string, go):
        g = sgf_generator(sgf_string, board_size=9, go=go)

        state, move, winner, score, comment = next(g)
        assert state.current_player == go.Color.BLACK
        assert move == (3, 4)
        assert winner == go.Color.BLACK
        assert score == 26
        assert comment == 'comment1'

        state, move, winner, score, comment = next(g)
        assert state.current_player == go.Color.WHITE
        assert move == go.Pass
        assert winner == go.Color.BLACK
        assert score == 26
        assert comment == 'comment2'

        state, move, winner, score, comment = next(g)
        assert state.current_player == go.Color.BLACK
        assert move == go.Pass
        assert winner == go.Color.BLACK
        assert score == 26
        assert comment == 'comment3'

        state, move, winner, score, comment = next(g)
        assert state.current_player == go.Color.WHITE
        assert move is None
        assert winner == go.Color.BLACK
        assert score == 26
        assert comment == ''

    def test_when_board_size_is_mismatch_then_raises_exception(
            self, sgf_string
    ):
        with pytest.raises(SgfContentError):
            for _ in sgf_generator(sgf_string, board_size=19, go=None):
                pass


def test_parse_migo():
    import migo.features
    import numpy as np
    sgf_string = """
(;GM[1]SZ[9]KM[7.0]RU[Chinese]
        AW[bb][cc]AB[dd][de][df]
)
"""
    initial_state, moves, winner, score = parse_one_game(
            sgf_string, board_size=9, go=migo,
            allow_ongoing_game=True
        )
    assert len(initial_state.history_buffer) == 0
    assert len(initial_state.empties) == (81 - 5)
    board = initial_state.board
    assert np.any(board)
    assert initial_state.current_player == migo.Color.BLACK
    board = migo.features.board_i(initial_state, 0)
    assert board[0].sum() == 3
    assert board[1].sum() == 2
    for i in range(initial_state.max_history_n):
        history = migo.features.history_n(initial_state, i)
        assert history.sum() == 5


def test_parse_cygo():
    sgf_string = """
(;GM[1]SZ[9]KM[7.0]RU[Chinese]
        AW[bb][cc]AB[dd][de][df]
)
"""
    import cygo.features
    import numpy as np
    initial_state, moves, winner, score = parse_one_game(
            sgf_string, board_size=9, go=cygo,
            allow_ongoing_game=True
        )
    assert initial_state.move_history == []
    assert initial_state.color_move_history(cygo.Color.BLACK) == []
    assert initial_state.color_move_history(cygo.Color.WHITE) == []
    for i in range(initial_state.max_history_n):
        history = cygo.features.history_n(initial_state, i)
        assert history.sum() == 5

    moves = initial_state.legal_moves()
    assert len(moves) == 81 - 5

    board_b = cygo.features.board_i(initial_state, 0, cygo.Color.BLACK)
    assert board_b.sum() == 3
    board_w = cygo.features.board_i(initial_state, 0, cygo.Color.WHITE)
    assert board_w.sum() == 2
