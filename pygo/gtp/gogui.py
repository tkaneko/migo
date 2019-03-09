import enum
from collections import OrderedDict, namedtuple
from typing import Callable, Tuple, Sequence

from pygo.gtp import Status, GTPRunner


class CommandType(str, enum.Enum):
    SBOARD = 'sboard'
    DBOARD = 'dboard'
    CBOARD = 'cboard'
    STRING = 'string'
    HSTRING = 'hstring'
    HPSTRING = 'hpstring'
    PSTRING = 'pstring'
    PLIST = 'plist'
    PARAM = 'param'
    PSPAIRS = 'pspairs'
    VARC = 'varc'
    GFX = 'gfx'
    NONE = 'none'


GoGuiParam = namedtuple('GoGuiParam', 'name type gogui_type value')


class GoGuiParams:

    def __init__(self, params: Sequence[GoGuiParam]):
        self.params = OrderedDict()

        for param in params:
            self.params[param.name] = (param.type, param.gogui_type, param.value)

    @property
    def param_names(self):
        return self.params.keys()

    def keys(self):
        return self.param_names

    def __getitem__(self, key):
        if key in self.params:
            return self.__getattr__(key)

        raise KeyError

    def __getattr__(self, name):
        if name in self.params:
            param_type, _, param_value = self.params[name]

            return param_type(param_value)

        raise AttributeError

    def update(self, key, value):
        assert key in self.params

        param_type, gogui_type, _ = self.params[key]

        self.params[key] = (param_type, gogui_type, value)

    def __call__(self, param_name=None, param_value=None):
        if param_name is None and param_value is None:
            return Status.success, str(self)

        self.update(param_name, param_value)

        return Status.success, ""

    def __str__(self):
        return '\n'.join(["[{type}] {param} {value}".format(type=gogui_type, param=param_name, value=value)
                          for (param_name, (param_type, gogui_type, value)) in self.params.items()])


class GoGuiGTPRunner(GTPRunner):

    def __init__(self):
        super().__init__()

        self._analyze_callbacks = []

        self.add_callback('gogui_analyze_commands', self.cmd_gogui_analyze_commands, arity=0)

    def add_analyze_callback(self,
                             command_type: CommandType,
                             command_str: str,
                             callback: Callable[..., Tuple[Status, str]],
                             check_arity=True,
                             display_name: str=None,
                             description: str=None) -> None:

        command_tokens = command_str.split()

        self._assert_command_tokens(command_tokens)
        self._analyze_callbacks.append("%s/%s/%s" % (command_type.value, display_name or command_str, command_str))

        arity = len(command_tokens) - 1 if check_arity else None

        if command_tokens[0] not in self.list_commands:
            self.add_callback(command_tokens[0], callback, arity=arity, description=description)

    def cmd_gogui_analyze_commands(self, *_) -> Tuple[Status, str]:
        return Status.success, "\n".join(self._analyze_callbacks)

    @staticmethod
    def _assert_command_tokens(command_tokens) -> None:
        assert len(command_tokens) > 0

        for param in command_tokens[1:]:
            assert param in {'%s', '%p', '%c', '%w', '%r'}
