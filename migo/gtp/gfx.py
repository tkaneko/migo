import enum
if hasattr(enum, 'StrEnum'):
    # Python 3.11+
    StrEnum = enum.StrEnum
else:
    import strenum
    StrEnum = strenum.StrEnum


class GFXType(StrEnum):
    influence = 'INFLUENCE'
    label = 'LABEL'
    variation = 'VAR'
    status = 'TEXT'
    color = 'COLOR'


class GFXPlayer(StrEnum):
    black = 'b'
    white = 'w'


class GFXSymbol(StrEnum):
    square = 'SQUARE'
    triangle = 'TRIANGLE'
    circle = 'CIRCLE'
    mark = 'MARK'
    white = 'WHITE'
    black = 'BLACK'


class GFX:

    def __init__(self):
        self._output = {}

    def set_influence(self, vertex: str, influence: float) -> None:
        self._output[GFXType.influence] = self._output.get(
            GFXType.influence, ''
        ) + "{} {} ".format(vertex, influence)

    def set_color(self, vertex: str, color: str) -> None:
        key = 'COLOR {}'.format(color)
        self._output[key] = self._output.get(key, '') + "{} ".format(vertex)

    def set_label(self, vertex: str, label: str) -> None:
        self._output[GFXType.label] = self._output.get(GFXType.label, '') \
            + "{} {} ".format(vertex, label)

    def add_variation(self, player: GFXPlayer, vertex: str) -> None:
        self._output[GFXType.variation] = self._output.get(
            GFXType.variation, '') + "{} {} ".format(player, vertex)

    def set_status(self, status: str) -> None:
        self._output[GFXType.status] = status

    def set_symbol(self, vertex: str, symbol: GFXSymbol) -> None:
        self._output[symbol] = self._output.get(symbol, '') \
            + "{vertex} ".format(vertex=vertex)

    def output(self) -> str:
        return "\n".join(["{} ".format(key) + "{}".format(value).strip()
                          for key, value in self._output.items()])
