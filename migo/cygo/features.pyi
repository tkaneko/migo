from __future__ import annotations
import migo.cygo
import numpy
__all__: list[str] = ['batch_features', 'batch_features_with_zone', 'board_i', 'collate', 'collate_ext', 'collatez', 'color_black', 'color_white', 'features_at', 'history_n', 'make_territory']
def batch_features(state_list: list[migo.cygo.State], history_n: int) -> tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int8]]:
    """
    Return a batch of features.
    """
def batch_features_with_zone(state_list: list[migo.cygo.State], history_n: int, zone: list[numpy.ndarray[numpy.int8]]) -> tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int8]]:
    """
    Return a batch of features.
    """
def board_i(state: migo.cygo.State, i: int, color: migo.cygo.Color = ...) -> numpy.ndarray[numpy.float32]:
    """
    Get the i-th before board feature. If c = Color.EMPTY, returns both color's features
    """
def collate(indices: numpy.ndarray[numpy.int32], history_n: int, board_size: int, move_offset: numpy.ndarray[numpy.int32], game_moves: numpy.ndarray[numpy.int16], winner: numpy.ndarray[numpy.int8], data_offset: numpy.ndarray[numpy.int32], ignore_opening_moves: int, correct_invalid_index: bool = False) -> tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int16], numpy.ndarray[numpy.int8]]:
    """
    collate function for SgfDataset, implemented in C++
    """
def collate_ext(indices: numpy.ndarray[numpy.int32], history_n: int, board_size: int, move_offset: numpy.ndarray[numpy.int32], game_moves: numpy.ndarray[numpy.int16], winner: numpy.ndarray[numpy.int8], data_offset: numpy.ndarray[numpy.int32], ignore_opening_moves: int, enabled_colors: list[int], aux_zones: numpy.ndarray[numpy.int8], aux_plane_labels: numpy.ndarray[numpy.int8], aux_values: numpy.ndarray[numpy.float32]) -> tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int16], numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int8], numpy.ndarray[numpy.float32]]:
    """
    collate function for ExtendedDataset, implemented in C++
    """
def collatez(indices: numpy.ndarray[numpy.int32], history_n: int, board_size: int, move_offset: numpy.ndarray[numpy.int32], game_moves: numpy.ndarray[numpy.int16], winner: numpy.ndarray[numpy.int8], data_offset: numpy.ndarray[numpy.int32], ignore_opening_moves: int, zones: numpy.ndarray[numpy.int8], zone_score: numpy.ndarray[numpy.float32], correct_invalid_index: bool = False) -> tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int16], numpy.ndarray[numpy.int8], numpy.ndarray[numpy.float32]]:
    """
    collate function for ZoneDataset, implemented in C++
    """
def color_black(arg0: migo.cygo.State) -> numpy.ndarray[numpy.float32]:
    ...
def color_white(arg0: migo.cygo.State) -> numpy.ndarray[numpy.float32]:
    ...
def features_at(board_size: int, moves: numpy.ndarray[numpy.int16], ids: numpy.ndarray[numpy.int32], history_n: int) -> numpy.ndarray[numpy.float32]:
    """
    Return a sequence of the set of features for given sequence of states specified by moves and ids.
    """
def history_n(state: migo.cygo.State, i: int, color: migo.cygo.Color = ...) -> numpy.ndarray[numpy.float32]:
    """
    Get the board history from n-th before. If c = Color.EMPTY, returns both color's features
    """
def make_territory(board_size: int, move_offset: numpy.ndarray[numpy.int32], game_moves: numpy.ndarray[numpy.int16]) -> numpy.ndarray[numpy.int8]:
    """
    compute territory at game end
    
    return ndarray with dimension games*(board_size**2 + 1), int8
    """
