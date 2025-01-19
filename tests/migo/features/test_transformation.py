import numpy as np
import numpy.testing as npt
import pytest

from migo.misc import move_to_index
from migo.features.transformation import THTransformation, index_to_onehot, onehot_to_index, transform_index, \
    rotate_features_and_action


class TestOneHot:

    @pytest.mark.parametrize('board_size, index, expected',
                             [
                                 (3, 0, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])),
                                 (3, 3, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])),
                                 (3, 8, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])),

                             ])
    def test_index_to_onehot(self, board_size, index, expected):
        npt.assert_array_equal(index_to_onehot(index, board_size), expected)

    def test_index_to_onehot_out_of_bound(self):
        board_size = 9

        with pytest.raises(AssertionError):
            index_to_onehot(-1, board_size)

        with pytest.raises(AssertionError):
            index_to_onehot(board_size ** 2, board_size)

    @pytest.mark.parametrize('board_size, plane, expected',
                             [
                                 (3, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 0),
                                 (3, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), 3),
                                 (3, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 8),
                             ]
                             )
    def test_onehot_to_index(self, board_size, plane, expected):
        npt.assert_array_equal(onehot_to_index(plane, board_size), expected)


class TestTHTransformation:

    def test_rotate_90(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[1, 0],
             [0, 0]],
            [[0, 0],
             [1, 0]]
        ])

        npt.assert_array_equal(THTransformation.rotate_90(planes), expected_planes)

    def test_rotate_180(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[0, 0],
             [1, 0]],
            [[0, 0],
             [0, 1]]
        ])

        npt.assert_array_equal(THTransformation.rotate_180(planes), expected_planes)

    def test_rotate_270(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[0, 0],
             [0, 1]],
            [[0, 1],
             [0, 0]]
        ])

        npt.assert_array_equal(THTransformation.rotate_270(planes), expected_planes)

    def test_fliplr(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[1, 0],
             [0, 0]],
            [[0, 1],
             [0, 0]]
        ])

        npt.assert_array_equal(THTransformation.fliplr(planes), expected_planes)

    def test_flipud(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[0, 0],
             [0, 1]],
            [[0, 0],
             [1, 0]]
        ])

        npt.assert_array_equal(THTransformation.flipud(planes), expected_planes)

    def test_diagonal_left_down(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[0, 1],
             [0, 0]],
            [[0, 0],
             [0, 1]]
        ])

        npt.assert_array_equal(THTransformation.diagonal_left_down(planes), expected_planes)

    def test_diagonal_right_down(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_planes = np.array([
            [[0, 0],
             [1, 0]],
            [[1, 0],
             [0, 0]]
        ])

        npt.assert_array_equal(THTransformation.diagonal_right_down(planes), expected_planes)

    def test_inverse(self):
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        npt.assert_array_equal(planes, THTransformation.rotate_90((THTransformation.rotate_270(planes))))
        npt.assert_array_equal(planes, THTransformation.rotate_180((THTransformation.rotate_180(planes))))
        npt.assert_array_equal(planes, THTransformation.rotate_270((THTransformation.rotate_90(planes))))

        npt.assert_array_equal(planes, THTransformation.fliplr((THTransformation.fliplr(planes))))
        npt.assert_array_equal(planes, THTransformation.flipud((THTransformation.flipud(planes))))
        npt.assert_array_equal(planes, THTransformation.diagonal_left_down((THTransformation.diagonal_left_down(planes))))
        npt.assert_array_equal(planes, THTransformation.diagonal_right_down((THTransformation.diagonal_right_down(planes))))

    def test_transform_index_rotate_90(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((8, 0), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.rotate_90, size)

        assert expected_action == actual_action

    def test_transform_index_rotate_180(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((8, 8), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.rotate_180, size)

        assert expected_action == actual_action

    def test_transform_index_rotate_270(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((0, 8), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.rotate_270, size)

        assert expected_action == actual_action

    def test_transform_index_fliplr(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((0, 8), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.fliplr, size)

        assert expected_action == actual_action

    def test_transform_index_flipud(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((8, 0), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.flipud, size)

        assert expected_action == actual_action

    def test_transform_index_diagonal_left_down(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((8, 8), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.diagonal_left_down, size)

        assert expected_action == actual_action

    def test_transform_index_diagonal_right_down(self):
        size = 9
        action = (0, 0)

        expected_action = move_to_index((0, 0), size)
        actual_action = transform_index(move_to_index(action, size), THTransformation.diagonal_right_down, size)

        assert expected_action == actual_action

    def test_transform_index_pass(self):
        size = 9
        actions = [-1, size ** size, None]

        for action in actions:
            with pytest.raises(AssertionError):
                transform_index(action, THTransformation.diagonal_right_down, size)

    def test_rotate_features_and_action_when_action_is_pass(self):
        action = -1
        planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        expected_action = 2 ** 2
        expected_planes = np.array([
            [[0, 1],
             [0, 0]],
            [[1, 0],
             [0, 0]]
        ])

        actual_plane, actual_action = rotate_features_and_action(planes, action, 2, THTransformation.identity)

        assert actual_action == expected_action
        npt.assert_array_equal(actual_plane, expected_planes)
