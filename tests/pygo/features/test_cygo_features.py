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

        expected = np.concatenate([expected_0, expected_1, expected_2] + [expected_outbound] * 7)

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

        npt.assert_array_equal(np.concatenate([expected_white, expected_black]), F.leela_board(state))

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

        expected_history = np.concatenate([black_0, white_0, black_1, white_1, black_2, white_2])

        for n in range(3, 10):
            outbounds = np.zeros((2 * (n - 2), 3, 3))
            expected = np.concatenate([expected_history, outbounds])

            assert expected.shape == (2 * (n + 1), 3, 3)

            npt.assert_array_equal(expected, F.history_n_bw(state, n))
