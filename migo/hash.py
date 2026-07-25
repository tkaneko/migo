from typing import Dict, List

import numpy as np

from migo.misc import Color, Coord, all_coordinates


class ZobristHash:
    """Implementation of Zobrist hashing for efficient board state representation."""

    __MAX_SIZE = 25
    __HASH_TABLE = None

    def __init__(self, hash_value=0):
        """Initialize the ZobristHash state with an optional initial value.

        :param hash_value: An optional initial hash value.
        """
        ZobristHash.__create_hash_table()

        self._hash_value = np.uint64(hash_value)

    @classmethod
    def __create_hash_table(cls) -> None:
        """Generate and cache the random bitstrings for each position and color."""
        if ZobristHash.__HASH_TABLE:
            return

        random_engine = np.random.RandomState(seed=0)

        cls.__HASH_TABLE = {
            Color.WHITE: random_engine.randint(
                np.iinfo(np.uint64).max,
                size=(cls.__MAX_SIZE, cls.__MAX_SIZE),
                dtype=np.uint64,
            ),
            Color.BLACK: random_engine.randint(
                np.iinfo(np.uint64).max,
                size=(cls.__MAX_SIZE, cls.__MAX_SIZE),
                dtype=np.uint64,
            ),
        }

    @property
    def value(self) -> np.uint64:
        """The hash value as an unsigned 64-bit integer.

        :return: The current hash value.
        """
        return self._hash_value

    def __str__(self) -> str:
        """Return the string representation of the hash value.

        :return: A string representation of the hash value.
        """
        return str(self._hash_value)

    def __repr__(self) -> str:
        """Return a formal string representation of the ZobristHash instance.

        :return: A formal string representation of the ZobristHash instance.
        """
        return 'ZobristHash(hash_value=%s)' % self._hash_value

    def update(self, position: Coord, color: Color) -> None:
        """Update the hash value by XORing it with the value at the given position and color.

        :param position: The coordinate to update.
        :param color: The color of the piece.
        """
        assert self.__HASH_TABLE
        self._hash_value = np.bitwise_xor(
            self._hash_value, self.__HASH_TABLE[color][position]
        )


class NeighborTable:
    """
    A lookup table that caches adjacent coordinates on a Go board to optimize
    the retrieval of neighbors during game tree searches and evaluation.
    """

    __4_NEIGHBORS_CACHE: Dict[int, Dict[Coord, List[Coord]]] = {}
    __DIAGONAL_NEIGHBORS_CACHE: Dict[int, Dict[Coord, List[Coord]]] = {}

    def __init__(self, size: int):
        """
        Initializes the neighbor table for a given board dimension.

        :param size: The side length of the Go board.
        """
        self.size = size

        self.__initialize_cache()

    def neighbors_crosswise(self, position: Coord) -> List[Coord]:
        """
        Returns the 4-way orthogonal neighbors (up, down, left, right)
        of a given intersection.

        :param position: The coordinate to query.
        :return: A list of adjacent cardinal coordinates.
        """
        return NeighborTable.__4_NEIGHBORS_CACHE[self.size][position]

    def neighbors_diagonal(self, position: Coord) -> List[Coord]:
        """
        Returns the 4-way diagonal neighbors of a given intersection.

        :param position: The coordinate to query.
        :return: A list of adjacent diagonal coordinates.
        """
        return NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size][position]

    def __initialize_cache(self) -> None:
        """Initialize the neighbor caches for the current board size."""

        def _on_board(position: Coord):
            row, col = position
            return 0 <= row < self.size and 0 <= col < self.size

        if self.size not in NeighborTable.__4_NEIGHBORS_CACHE:
            NeighborTable.__4_NEIGHBORS_CACHE[self.size] = {}

            for x, y in all_coordinates(self.size):
                neighbors = [
                    xy
                    for xy in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    if _on_board(xy)
                ]
                NeighborTable.__4_NEIGHBORS_CACHE[self.size][(x, y)] = neighbors

        if self.size not in NeighborTable.__DIAGONAL_NEIGHBORS_CACHE:
            NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size] = {}

            for x, y in all_coordinates(self.size):
                neighbors = [
                    xy
                    for xy in [
                        (x - 1, y - 1),
                        (x - 1, y + 1),
                        (x + 1, y - 1),
                        (x + 1, y + 1),
                    ]
                    if _on_board(xy)
                ]

                NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size][(x, y)] = (
                    neighbors
                )
