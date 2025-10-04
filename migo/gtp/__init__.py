"""
game playing via `gtp`

- http://www.lysator.liu.se/~gunnar/gtp/
- https://www.gnu.org/software/gnugo/gnugo_19.html
"""
from migo.gtp.gtp import GTPRunner
from migo.gtp.gtp import GTPRuntimeError
from migo.gtp.gtp import Status

from migo.gtp.utils import move_to_str
from migo.gtp.utils import parse_color
from migo.gtp.utils import parse_move

from .gogui import GoGuiGTPRunner, GoGuiParam, GoGuiParams, CommandType


__all__ = [
    'GTPRunner', 'GTPRuntimeError', 'Status',
    'move_to_str', 'parse_color', 'parse_move',
    'GoGuiGTPRunner', 'GoGuiParam', 'GoGuiParams', 'CommandType',
]
