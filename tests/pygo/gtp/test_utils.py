import pytest

from migo import Color, Pass
from migo.gtp import parse_color, parse_move, move_to_str


class TestGtpUtils:

    def test_parse_color_when_argument_is_valid_then_returns_color(self):
        assert Color.BLACK == parse_color('b')
        assert Color.BLACK == parse_color('B')

        assert Color.WHITE == parse_color('w')
        assert Color.WHITE == parse_color('W')

    def test_parse_color_when_argument_is_invalid_then_returns_none(self):
        with pytest.raises(ValueError):
            parse_color('invalid')

        with pytest.raises(ValueError):
            parse_color('bAAAAAA')

        with pytest.raises(ValueError):
            parse_color('BAAAAAA')

        with pytest.raises(ValueError):
            parse_color('wAAAAAA')

        with pytest.raises(ValueError):
            parse_color('WAAAAAA')

    def test_parse_move_when_argument_is_valid_then_returns_move(self):
        assert (0, 0) == parse_move('a1')
        assert (0, 0) == parse_move('A1')
        assert (18, 18) == parse_move('t19')
        assert (18, 18) == parse_move('T19')

        assert parse_move('pass') is Pass
        assert parse_move('PASS') is Pass

    def test_parse_move_when_argument_is_invalid_then_raises(self):
        with pytest.raises(ValueError):
            parse_move('A26')

        with pytest.raises(ValueError):
            parse_move('')

        with pytest.raises(ValueError):
            parse_move('A 1')

        with pytest.raises(ValueError):
            parse_move('invalid text')

    def test_move_to_str_when_argument_is_valid_then_returns_str(self):
        assert move_to_str((0, 0)) == 'A1'
        assert move_to_str((18, 18)) == 'T19'
        assert move_to_str(Pass) == 'PASS'

    def test_move_to_str_when_argument_is_invalid_then_raises(self):
        with pytest.raises(ValueError):
            move_to_str((-1, -1))

        with pytest.raises(ValueError):
            move_to_str((25, 25))

        with pytest.raises(ValueError):
            move_to_str([0, 0])
