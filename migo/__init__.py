from .misc import Color, Coord, IllegalMoveError, Move, Pass, all_coordinates
from .state import State
from .network import PVNetwork, ExtendedNetwork, load_network
from .utility.model import Node, eval_state_by_model
from .record import SimpleRecord, record_to_sgf, parse_sgf_game
from .dataset import SgfDataset, ZoneDataset, ExtendedDataset, load_dataset

__all__ = [
    'Color', 'Coord', 'IllegalMoveError', 'Move', 'Pass', 'all_coordinates',
    'State',
    'PVNetwork', 'ExtendedNetwork', 'load_network',
    'Node', 'eval_state_by_model',
    'SimpleRecord', 'record_to_sgf', 'parse_sgf_game',
    'SgfDataset', 'ZoneDataset', 'ExtendedDataset', 'load_dataset',
]


def version():
    import importlib.metadata
    return importlib.metadata.version('migo')
