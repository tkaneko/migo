import random

import pytest

from pygo import Color, all_coordinates
from pygo.hash import ZobristHash, NeighborTable


@pytest.mark.parametrize("color", [Color.BLACK, Color.WHITE])
def test_zobrist_hash(color):
    moves1 = list(all_coordinates(size=9))
    moves2 = list(all_coordinates(size=9))

    random.shuffle(moves1)
    random.shuffle(moves2)

    hash1 = ZobristHash()
    hash2 = ZobristHash()

    for move1, move2 in zip(moves1, moves2):
        hash1.update(position=move1, color=color)
        hash2.update(position=move2, color=color)

    assert hash1.value == hash2.value


@pytest.fixture(scope='module', autouse=True)
def table():
    table = NeighborTable(size=9)

    yield table


def test_neighbors_crosswise(table):
    point = (5, 5)
    expected = [(4, 5), (5, 4), (5, 6), (6, 5)]
    actual = table.neighbors_crosswise(point)

    assert len(expected) == len(actual)


def test_neighbors_crosswise_when_corner_given_then_returns_two_elements_only(table):
    point = (0, 0)
    expected = [(1, 0), (0, 1)]
    actual = table.neighbors_crosswise(point)

    assert len(expected) == len(actual)


def test_neighbors_diagonal(table):
    point = (5, 5)
    expected = [(4, 4), (6, 4), (4, 6), (6, 6)]

    actual = table.neighbors_diagonal(point)

    assert len(expected) == len(actual)


def test_neighbors_diagonal_when_corner_given_then_returns_one_element_only(table):
    point = (0, 0)
    expected = [(1, 1)]
    actual = table.neighbors_diagonal(point)

    assert len(expected) == len(actual)
