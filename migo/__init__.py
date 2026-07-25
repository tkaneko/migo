from .dataset import ExtendedDataset, SgfDataset, load_dataset
from .misc import Color, Coord, IllegalMoveError, Move, Pass, all_coordinates
from .network import ExtendedNetwork, PVNetwork, load_network
from .record import SimpleRecord, parse_sgf_game, record_to_sgf
from .state import State
from .utility.model import Node, eval_state_by_model

__all__ = [
    'Color',
    'Coord',
    'IllegalMoveError',
    'Move',
    'Pass',
    'all_coordinates',
    'State',
    'PVNetwork',
    'ExtendedNetwork',
    'load_network',
    'Node',
    'eval_state_by_model',
    'SimpleRecord',
    'record_to_sgf',
    'parse_sgf_game',
    'SgfDataset',
    'ExtendedDataset',
    'load_dataset',
]


def version():
    import importlib.metadata

    return importlib.metadata.version('migo')
