import numpy as np
import pygo.features
import cygo


def check_consistency(pstate, cstate, n=3):
    xpb = pygo.features.board(pstate)
    xcb = pygo.features.board(cstate)
    assert np.array_equal(xpb, xcb)
    xpb = pygo.features.pygo.board(pstate)
    assert np.array_equal(xpb, xcb)

    xpc = pygo.features.color(pstate)
    xcc = pygo.features.color(cstate)
    assert np.array_equal(xpc, xcc)

    xpcb = pygo.features.color_black(pstate)
    xccb = pygo.features.color_black(cstate)
    assert np.array_equal(xpcb, xccb)

    xpcw = pygo.features.color_white(pstate)
    xccw = pygo.features.color_white(cstate)
    assert np.array_equal(xpcw, xccw)

    xplb = pygo.features.leela_board(pstate, n)
    xclb = pygo.features.leela_board(pstate, n)
    assert np.array_equal(xplb, xclb)

    xplc = pygo.features.leela_color(pstate)
    xclc = pygo.features.leela_color(pstate)
    assert np.array_equal(xplc, xclc)

    xpo = pygo.features.ones(pstate)
    xco = pygo.features.ones(cstate)
    assert np.array_equal(xpo, xco)

    xpz = pygo.features.zeros(pstate)
    xcz = pygo.features.zeros(cstate)
    assert np.array_equal(xpz, xcz)

    for i in range(pstate.max_history_n):
        xp = pygo.features.board_i(pstate, i)
        xc = pygo.features.board_i(cstate, i)
        assert np.array_equal(xp, xc)

    xph = pygo.features.history_n(pstate, n)
    xch = pygo.features.history_n(cstate, n)
    assert xph.shape == xch.shape
    assert np.array_equal(xph, xch)
    xph = pygo.features.pygo.history_n(pstate, n)
    assert np.array_equal(xph, xch)


def test_consistent():
    pstate = pygo.State(6)
    cstate = cygo.State(6)
    n = 3
    check_consistency(pstate, cstate, n)

    pstate.make_move((1, 2))
    cstate.make_move((1, 2))
    assert pstate.current_player == pygo.Color.WHITE
    assert cstate.current_player == cygo.Color.WHITE

    check_consistency(pstate, cstate, n)

    pstate.make_move((4, 4))
    cstate.make_move((4, 4))
    assert pstate.current_player == pygo.Color.BLACK
    assert cstate.current_player == cygo.Color.BLACK

    check_consistency(pstate, cstate, n)

    pstate.make_move(pygo.misc.Pass)  # None
    cstate.make_move(pygo.misc.Pass)  # None
    assert pstate.current_player == pygo.Color.WHITE
    assert cstate.current_player == cygo.Color.WHITE

    pstate.make_move((3, 4))
    cstate.make_move((3, 4))
    assert pstate.current_player == pygo.Color.BLACK
    assert cstate.current_player == cygo.Color.BLACK

    check_consistency(pstate, cstate, n)


def test_consistent_pass2():
    s = pygo.State(9)
    s.make_move((1, 1))
    s.make_move((1, 2))
    x = pygo.features.history_n(s, 4)  # history: (1, 1), (1, 2)
    s.make_move(None)
    s.make_move(None)
    y = pygo.features.history_n(s, 4)   # history: (1, 1), (1, 2), pass, pass
    z = pygo.features.pygo.history_n(s, 4)
    assert not np.array_equal(x, y)
    assert not np.array_equal(x, z)
    assert np.array_equal(y, z)

    cs = cygo.State(9)
    cs.make_move((1, 1))
    cs.make_move((1, 2))
    cx = pygo.features.history_n(cs, 4)  # history: (1, 1), (1, 2)
    cs.make_move(None)
    cs.make_move(None)
    cy = pygo.features.history_n(cs, 4)   # history: (1, 1), (1, 2), pass, pass

    assert np.array_equal(x, cx)
    assert np.array_equal(y, cy)
    assert np.array_equal(z, cy)
