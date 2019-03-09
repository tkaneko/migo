import io

import pytest

from pygo.gtp import GTPRuntimeError, GTPRunner, Status


class TestGTPRunner:

    @pytest.fixture(scope='class', autouse=True)
    def gtp_runner(self):
        def f_0():
            return Status.success, ""

        def f_1(_):
            return Status.success, ""

        def runtime_error():
            raise GTPRuntimeError

        runner = GTPRunner()

        runner.add_callback('arity_0', f_0, arity=0)
        runner.add_callback('arity_1', f_1, arity=1)
        runner.add_callback('runtime_error', runtime_error, arity=0)

        yield runner

    def test_init(self, gtp_runner):
        assert 'quit' in gtp_runner._callbacks
        assert 'list_commands' in gtp_runner._callbacks
        assert 'help' in gtp_runner._callbacks
        assert 'protocol_version' in gtp_runner._callbacks

    def test_quit(self, gtp_runner):
        assert (Status.quit, 'bye') == gtp_runner.execute_one_command('quit')

    def test_protocol_version(self, gtp_runner):
        assert (Status.success, '2') == gtp_runner.execute_one_command('protocol_version')

    def test_execute(self, gtp_runner):
        input_stream = io.StringIO("protocol_version\nquit")
        output_stream = io.StringIO()

        gtp_runner.execute(input_stream, output_stream)

    def test_execute_one_command_when_line_is_empty(self, gtp_runner):
        assert (Status.noop, "") == gtp_runner.execute_one_command("")
        assert (Status.noop, "") == gtp_runner.execute_one_command(" ")
        assert (Status.noop, "") == gtp_runner.execute_one_command("\n")

    def test_execute_one_command_when_arity_is_wrong(self, gtp_runner):
        assert Status.failure == gtp_runner.execute_one_command("arity_0 x")[0]
        assert Status.failure == gtp_runner.execute_one_command("arity_1")[0]
        assert Status.failure == gtp_runner.execute_one_command("arity_1 ")[0]

    def test_execute_one_command_when_callback_raises_an_exception(self, gtp_runner):
        assert Status.failure == gtp_runner.execute_one_command("runtime_error")[0]
