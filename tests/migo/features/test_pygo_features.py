import numpy as np
import numpy.testing as npt
import pytest

from migo import State, Color, all_coordinates
from migo.features.utils import Order
from migo.features import migo as F
from migo.state import parse


class TestFeatures:

    def test_stone_color(self):
        state, _ = parse("B B B|"
                         "W W .|"
                         ". . .|")

        expected_black = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        expected_white = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        expected_empty = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])

        planes = F.stone_color(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 3)

        npt.assert_array_equal(expected_black, planes[:, :, 0])
        npt.assert_array_equal(expected_white, planes[:, :, 1])
        npt.assert_array_equal(expected_empty, planes[:, :, 2])

    def test_turns_since(self):
        size = 3
        state = State(board_size=size)

        expected_planes = [
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        ]

        for move in all_coordinates(size=size):
            state.make_move(move, Color.BLACK)

        planes = F.turns_since(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 8)

        for i, expected_plane in enumerate(expected_planes):
            npt.assert_array_equal(expected_plane, planes[:, :, i])

    def test_liberties(self):
        state, _ = parse("B W W|"
                         ". B W|"
                         ". . W|")

        expected_plane1 = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
        expected_plane2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        planes = F.liberties(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 8)

        npt.assert_array_equal(expected_plane1, planes[:, :, 0])
        npt.assert_array_equal(expected_plane2, planes[:, :, 1])

    def test_capture_size(self):
        state, moves = parse("B W W W B .|"
                             "B W W W B .|"
                             "1 W W W B .|"
                             ". B B B B .|"
                             ". . . B W B|"
                             ". . . B W 2|")

        expected_planes = [
            np.array([[0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0]]),
            np.zeros((6, 6)),
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]]),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        ]

        planes = F.capture_size(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 8)

        for i, expected_plane in enumerate(expected_planes):
            npt.assert_array_equal(expected_plane, planes[:, :, i])

    def test_self_atari_size(self):
        state, _ = parse(". . 4 B W .|"
                         "W B B W W .|"
                         ". W W W . .|"
                         ". W . . . .|"
                         "W . . . . W|"
                         ". B W . . .|")

        expected_planes = [
            np.array([[1, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]]),
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0]]),
            np.zeros((6, 6)),
            np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
            np.zeros((6, 6)),
        ]

        planes = F.self_atari_size(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 8)

        for i, expected_plane in enumerate(expected_planes):
            npt.assert_array_equal(expected_plane, planes[:, :, i])

    def test_liberties_after_move(self):
        state, _ = parse("2 1 W W W B|"
                         "2 W W W 2 2|"
                         "3 3 7 3 4 3|"
                         "3 8 B 8 6 3|"
                         "3 7 B 6 B 4|"
                         "7 B B B 8 B|", next_color=Color.BLACK)

        expected_planes = [
            # 1 liberty
            np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),

            # 2 liberties
            np.array([[1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),

            # 3 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),

            # 4 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]]),

            # 5 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),

            # 6 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),

            # 7 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0]]),

            # >8 liberties
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0]]),
        ]

        planes = F.liberties_after_move(state, order=Order.HWC)

        assert planes.shape == (state.board_size, state.board_size, 8)

        for i, expected_plane in enumerate(expected_planes):
            npt.assert_array_equal(expected_plane, planes[:, :, i])

    def test_sensibleness(self):
        state, _ = parse(". . . . W .|"
                         ". . B B . W|"
                         ". B . B . .|"
                         ". B B B . .|"
                         ". B . B . .|"
                         ". B B . . .|", next_color=Color.BLACK)

        expected_plane = np.array([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 1]
        ])

        plane = F.sensibleness(state, order=Order.HWC)

        assert plane.shape == (state.board_size, state.board_size, 1)

        npt.assert_array_equal(expected_plane, plane[:, :, 0])


class TestHistoryFeature:
    def test_board_i(self):
        state = State(board_size=3, max_history_n=10)

        state.make_move((0, 0))  # Black
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

        assert state.current_player == Color.WHITE
        assert len(state.history_buffer) == 3

        npt.assert_array_equal(expected_2, F.board_i(state, 2, order=Order.CHW))
        npt.assert_array_equal(expected_1, F.board_i(state, 1, order=Order.CHW))
        npt.assert_array_equal(expected_0, F.board_i(state, 0, order=Order.CHW))

        for n in range(3, 10):
            npt.assert_array_equal(expected_outbound, F.board_i(state, n, order=Order.CHW))

    def test_board_i_when_state_has_no_history(self):
        state = State(max_history_n=0)

        # with no exception when i = 0
        F.board_i(state, i=0, order=Order.HWC)

        with pytest.raises(AssertionError):
            F.board_i(state, i=1, order=Order.HWC)


class TestLeelaFeature:

    def test_leela_board_initial_state(self):
        state = State(3, max_history_n=8)

        npt.assert_equal(F.leela_board(state, order=Order.CHW, n=2),
                         np.array([
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                         ]))

    def test_leela_board_with_history(self):
        state = State(3, max_history_n=8)

        moves = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]

        for move in moves:
            state.make_move(move, Color.BLACK)

        state.current_player = Color.BLACK

        expected_black_history = np.array(
            [
                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],

                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 0]],

                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 0, 0]],

                [[1, 1, 1],
                 [1, 1, 1],
                 [0, 0, 0]],

                [[1, 1, 1],
                 [1, 1, 0],
                 [0, 0, 0]],

                [[1, 1, 1],
                 [1, 0, 0],
                 [0, 0, 0]],

                [[1, 1, 1],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[1, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
            ],
        )

        expected_planes = np.concatenate((expected_black_history, np.zeros((8, 3, 3))))
        actual_planes = F.leela_board(state, Order.CHW)

        npt.assert_equal(expected_planes, actual_planes)
