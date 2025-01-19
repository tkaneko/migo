import enum
import sys
from collections import namedtuple

from logging import getLogger
from typing import Callable, Tuple, List


Status = enum.Enum('Status', 'success failure quit noop')


Callback = namedtuple('Callback', 'f arity description')


class GTPRuntimeError(Exception):
    pass


class GTPRunner:

    def __init__(self, *, logger=None):
        self._logger = logger or getLogger(__name__)

        self._callbacks = {}  # type: Dict[str, Callback]

        self.add_callback('quit', self.cmd_quit, arity=0)
        self.add_callback('list_commands', self.cmd_list_commands, arity=0)
        self.add_callback('help', self.cmd_help, arity=0)

        self.add_static_callback('protocol_version', '2')

    def add_callback(self, name: str, f: Callable[..., Tuple[Status, str]],
                     arity: int | None = None, description: str | None = None
                     ) -> None:
        self._logger.debug("Add callback '%s' (arity: %s, description: %s)"
                           % (name, arity, description))

        if arity is not None and arity < 0:
            raise ValueError("arity should be greater than or equal to 0,"
                             " which is %d" % arity)

        if name in self._callbacks:
            raise ValueError("callback named `%s` is already registered"
                             % name)

        self._callbacks[name] = Callback(f, arity, description)

    def add_static_callback(self, name: str, value: str) -> None:
        self.add_callback(name, lambda *args: (Status.success, value), arity=0)

    def execute_one_command(self, line: str) -> Tuple[Status, str]:
        words = line.split()

        if not words:
            return Status.noop, ""

        command = words[0]
        params = words[1:]

        if command not in self._callbacks:
            return self.cmd_unknown_command(command)

        callback = self._callbacks[command]

        if callback.arity is not None and len(params) != callback.arity:
            return Status.failure, "Callback `%s` required %d argument(s)," \
                " but provided %d argument(s)" \
                % (command, callback.arity, len(params))

        try:
            self._logger.debug("execute command '%s' with arguments %s"
                               % (command, params))

            if callback.arity == 0:
                return callback.f()
            else:
                return callback.f(*params)

        except GTPRuntimeError as e:
            if callback.description is None:
                return Status.failure, "Internal error occurred.\n{}".format(e)
            else:
                return Status.failure, "Internal error occurred." \
                    "\n{}\nusage: {}".format(e, callback.description)

    def execute(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout

        while True:
            line = stdin.readline()

            self._logger.debug("process line: '%s'" % line.rstrip())

            status, output = self.execute_one_command(line)

            if status == Status.noop:
                continue

            symbol = '?' if status == Status.failure else '='

            stdout.write("{symbol} {output}\n\n".format(symbol=symbol,
                                                        output=output))
            stdout.flush()

            if status == Status.quit:
                break

    @property
    def list_commands(self) -> List[str]:
        return [command for command in self._callbacks.keys()]

    def cmd_list_commands(self, *_) -> Tuple[Status, str]:
        return Status.success, "\n".join(self.list_commands)

    def cmd_known_commands(self, *_) -> Tuple[Status, str]:
        return self.cmd_list_commands(_)

    def cmd_help(self, *_) -> Tuple[Status, str]:
        return self.cmd_list_commands(_)

    @staticmethod
    def cmd_quit(*_) -> Tuple[Status, str]:
        return Status.quit, "bye"

    @staticmethod
    def cmd_unknown_command(command) -> Tuple[Status, str]:
        return Status.failure, "unknown command: %s" % command
