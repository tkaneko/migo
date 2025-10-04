import numpy as np
import numpy.testing as npt

from cygo import State, Color
from migo.features import cygo as F


class TestCyFeature:
    def test_board_i(self):
        state = State(board_size=3, max_history_n=10)

        state.make_move((0, 0))
        state.make_move((0, 1))  # White
        state.make_move((0, 2))  # Black

        expected_2 = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # White board
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # Black board
        ])

        expected_1 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # ditto
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        expected_0 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 1], [0, 0, 0], [0, 0, 0]],
        ])

        expected_outbound = np.zeros((2, 3, 3))

        npt.assert_array_equal(expected_2, F.board_i(state, 2))
        npt.assert_array_equal(expected_1, F.board_i(state, 1))
        npt.assert_array_equal(expected_0, F.board_i(state, 0))

        for i in range(3, 10):
            npt.assert_array_equal(expected_outbound, F.board_i(state, i))

    def test_history_n(self):
        state = State(board_size=3, max_history_n=10)

        state.make_move((0, 0))
        state.make_move((0, 1))  # White
        state.make_move((0, 2))  # Black

        expected_2 = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # White board
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # Black board
        ])

        expected_1 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # ditto
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        expected_0 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 1], [0, 0, 0], [0, 0, 0]],
        ])

        expected_outbound = np.zeros((2, 3, 3))

        expected = np.concatenate([expected_0, expected_1, expected_2]
                                  + [expected_outbound] * 7)

        npt.assert_array_equal(expected, F.history_n(state, 9))


class TestCyLeelaZeroFeature:
    def test_leela_board(self):
        state = State(board_size=3, max_history_n=8)

        state.make_move((0, 0))
        state.make_move((0, 1))  # White
        state.make_move((0, 2))  # Black

        expected_black = np.array([
            [[1, 0, 1], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        expected_white = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        npt.assert_array_equal(
            np.concatenate([expected_white, expected_black]),
            F.leela_board(state)
        )

    def test_leela_color_black(self):
        state = State(board_size=3)

        expected = np.concatenate([
            np.ones((1, 3, 3), dtype=np.float32),
            np.zeros((1, 3, 3), dtype=np.float32),
        ])

        npt.assert_array_equal(expected, F.leela_color(state))

    def test_leela_color_white(self):
        state = State(board_size=3)

        state.current_player = Color.WHITE

        expected = np.concatenate([
            np.zeros((1, 3, 3), dtype=np.float32),
            np.ones((1, 3, 3), dtype=np.float32),
        ])

        npt.assert_array_equal(expected, F.leela_color(state))


class TestCyFeatureBW:
    def test_history_n_bw(self):
        state = State(board_size=3, max_history_n=10)

        state.make_move((0, 0))
        state.make_move((0, 1))
        state.make_move((0, 2))

        black_0 = np.array([
            [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
        ])

        black_1 = np.array([
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])

        black_2 = np.array([
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])  # same as black_1

        white_0 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        ])

        white_1 = np.array([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        ])  # same as white_0

        white_2 = np.zeros(white_0.shape)

        expected_history = np.concatenate(
            [black_0, white_0, black_1, white_1, black_2, white_2]
        )

        for n in range(3, 10):
            outbounds = np.zeros((2 * (n - 2), 3, 3))
            expected = np.concatenate([expected_history, outbounds])

            assert expected.shape == (2 * (n + 1), 3, 3)

            npt.assert_array_equal(expected, F.history_n_bw(state, n))


def test_std_features():
    import cygo
    import migo
    import migo.record
    sgf_str = \
        '(;GM[1]SZ[9]KM[7.0]GN[R=8,B+R,81]RE[B+2.0]RU[Chinese];' \
        'B[ee];W[ec];B[cd];W[ce];B[dd];W[ed];B[de];W[cf];B[ge];W[gc];B[dg];' \
        'W[df];B[ef];W[cg];B[eh];W[dh];B[eg];W[di];B[hd];W[hc];B[db];W[gd];' \
        'B[he];W[eb];B[be];W[bf];B[bh];W[ch];B[ae];W[bi];B[ei];W[dc];B[cc];' \
        'W[cb];B[bb];W[da];B[ic];W[ib];B[id];W[ah];B[hb];W[gb];B[ia];W[ba];' \
        'B[bc];W[fe];B[ff];W[fd];B[af];W[ab];B[ib];W[ca];B[ac];W[ga];B[aa];' \
        'W[gf];B[hf];W[gg];B[hg];W[gh];B[hh];W[hi];B[gi];W[fi];B[ab];W[ag];' \
        'B[ih];W[ha];B[fh];W[bg];B[gi];W[bd];B[ad];W[fi];B[ea];W[fa];B[gi];' \
        'W[if];B[fg];W[fb];B[ig];W[gh];B[ii];W[gf];B[gg];W[tt];B[ie];W[tt];' \
        'B[tt])'
    game = migo.record.parse_sgf_game(sgf_str)
    history_n = 8
    move_ids = [37, 43, 3]
    xhs = []
    for id in move_ids:
        state = cygo.State(game.board_size, max_history_n=history_n)
        cygo.apply_moves(state, game.moves[:id])
        assert history_n == state.max_history_n
        xh = migo.features.history_n(state, history_n)
        xc = migo.features.color(state)
        assert xh.shape == (2*(history_n+1), game.board_size, game.board_size)
        if id >= history_n:
            for i in range(xh.shape[1]):
                assert xh[i].any(), f'plane {i} is filled by zeros for {id=}'
        xhs.append(xh)
        xhs.append(xc)
    xh = np.vstack(xhs)

    ch = cygo.features.features_at(
        game.board_size, game.moves, move_ids, history_n
    )
    assert xh.shape == ch.shape
    assert not np.array_equal(xh, np.zeros_like(xh))
    assert np.array_equal(xh, ch)


def test_batch_features():
    import cygo
    _, empties = cygo.features.batch_features([cygo.State(9)], 0)
    assert sum(empties[0]) == 82
    _, empties = cygo.features.batch_features([cygo.State(9)], 1)
    assert sum(empties[0]) == 82

    state = cygo.State(9)
    state.make_move((1, 2))
    state.make_move((3, 4))
    _, empties = cygo.features.batch_features([state], 1)
    assert sum(empties[0]) == 80


def test_batch_features_with_zone():
    import cygo
    zone = np.ones((9, 9), dtype=np.int8)
    x, empties = cygo.features.batch_features_with_zone(
        [cygo.State(9)], 0, [zone]
    )
    assert sum(empties[0]) == 82
    assert np.array_equal(x[0][3], zone)
    x, empties = cygo.features.batch_features_with_zone(
        [cygo.State(9)], 1, [zone]
    )
    assert sum(empties[0]) == 82
    assert np.array_equal(x[0][5], zone)

    state = cygo.State(9)
    state.make_move((1, 2))
    state.make_move((3, 4))
    x, empties = cygo.features.batch_features_with_zone([state], 1, [zone])
    assert sum(empties[0]) == 80
    assert np.array_equal(x[0][5], zone)
