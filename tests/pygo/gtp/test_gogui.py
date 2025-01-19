import pytest

from migo.gtp import Status
from migo.gtp.gogui import GoGuiGTPRunner, CommandType, GoGuiParam, GoGuiParams


class TestGoGuiGTPRunner:

    @pytest.fixture(scope='class', autouse=True)
    def gtp_runner(self):
        runner = GoGuiGTPRunner()

        yield runner

    def test_init(self, gtp_runner):
        assert 'gogui_analyze_commands' in gtp_runner._callbacks

    def test_add_analyze_command(self, gtp_runner):
        gtp_runner.add_analyze_callback(CommandType.NONE, 'command', lambda *args: (Status.success, ""))

        assert 'command' in gtp_runner.list_commands

    def test_add_analyze_command_command_str_is_invalid_then_raises_assertion_error(self, gtp_runner):
        def f():
            return Status.success, ""

        with pytest.raises(AssertionError):
            gtp_runner.add_analyze_callback(CommandType.NONE, '', f)

        with pytest.raises(AssertionError):
            gtp_runner.add_analyze_callback(CommandType.NONE, 'command %a', f)


class TestGoGuiParam:

    def test_init(self):
        params = GoGuiParams([
            GoGuiParam('param1', bool, 'bool', 1),
            GoGuiParam('param2', str, 'string', 'foo')
        ])

        expected_str = '[bool] param1 1\n[string] param2 foo'

        assert 'param1' in params.param_names
        assert 'param2' in params.param_names

        assert params.param1
        assert params.param2 == 'foo'

        assert expected_str == str(params)
        assert (Status.success, expected_str) == params()

    def test_update(self):
        params = GoGuiParams([
            GoGuiParam('param', bool, 'bool', 0)
        ])

        assert params.param == 0

        params('param', 1)

        expected_str = '[bool] param 1'

        assert params.param == 1
        assert expected_str == str(params)
        assert (Status.success, expected_str) == params()
