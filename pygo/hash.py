import numpy as np
from typing import Dict, List

from pygo.misc import Color, Coord, all_coordinates


class ZobristHash:
    __MAX_SIZE = 25
    __HASH_TABLE = None

    def __init__(self, hash_value=0):
        ZobristHash.__create_hash_table()

        self._hash_value = np.uint64(hash_value)

    @classmethod
    def __create_hash_table(cls):
        if ZobristHash.__HASH_TABLE:
            return

        random_engine = np.random.RandomState(seed=0)

        cls.__HASH_TABLE = {
            Color.WHITE:
                random_engine.randint(np.iinfo(np.uint64).max,
                                      size=(cls.__MAX_SIZE, cls.__MAX_SIZE),
                                      dtype=np.uint64),
            Color.BLACK:
                random_engine.randint(np.iinfo(np.uint64).max,
                                      size=(cls.__MAX_SIZE, cls.__MAX_SIZE),
                                      dtype=np.uint64)
        }

    @property
    def value(self) -> np.uint64:
        """hash value in np.uint64"""
        return self._hash_value

    def __str__(self):
        return str(self._hash_value)

    def __repr__(self):
        return "ZobristHash(hash_value=%s)" % self._hash_value

    def update(self, position: Coord, color: Color):
        self._hash_value = np.bitwise_xor(self._hash_value, self.__HASH_TABLE[color][position])


class NeighborTable:
    __4_NEIGHBORS_CACHE = {}  # type: Dict[int, Dict[Coord, List[Coord]]]
    __DIAGONAL_NEIGHBORS_CACHE = {}  # type: Dict[int, Dict[Coord, List[Coord]]]

    def __init__(self, size: int):
        self.size = size

        self.__initialize_cache()

    def neighbors_crosswise(self, position: Coord):
        return NeighborTable.__4_NEIGHBORS_CACHE[self.size][position]

    def neighbors_diagonal(self, position: Coord):
        return NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size][position]

    def __initialize_cache(self):
        def _on_board(position: Coord):
            row, col = position
            return 0 <= row < self.size and 0 <= col < self.size

        if self.size not in NeighborTable.__4_NEIGHBORS_CACHE:
            NeighborTable.__4_NEIGHBORS_CACHE[self.size] = {}

            for x, y in all_coordinates(self.size):
                neighbors = [xy for xy in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] if _on_board(xy)]
                NeighborTable.__4_NEIGHBORS_CACHE[self.size][(x, y)] = neighbors

        if self.size not in NeighborTable.__DIAGONAL_NEIGHBORS_CACHE:
            NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size] = {}

            for x, y in all_coordinates(self.size):
                neighbors = [xy for xy in [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
                             if _on_board(xy)]

                NeighborTable.__DIAGONAL_NEIGHBORS_CACHE[self.size][(x, y)] = neighbors
