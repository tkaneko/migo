import numpy as np
import migo.features
import cygo


def check_consistency(pstate, cstate, n=3):
    xpb = migo.features.board(pstate)
    xcb = migo.features.board(cstate)
    assert np.array_equal(xpb, xcb)
    xpb = migo.features.migo.board(pstate)
    assert np.array_equal(xpb, xcb)

    xpc = migo.features.color(pstate)
    xcc = migo.features.color(cstate)
    assert np.array_equal(xpc, xcc)

    xpcb = migo.features.color_black(pstate)
    xccb = migo.features.color_black(cstate)
    assert np.array_equal(xpcb, xccb)

    xpcw = migo.features.color_white(pstate)
    xccw = migo.features.color_white(cstate)
    assert np.array_equal(xpcw, xccw)

    xplb = migo.features.leela_board(pstate, n)
    xclb = migo.features.leela_board(pstate, n)
    assert np.array_equal(xplb, xclb)

    xplc = migo.features.leela_color(pstate)
    xclc = migo.features.leela_color(pstate)
    assert np.array_equal(xplc, xclc)

    xpo = migo.features.ones(pstate)
    xco = migo.features.ones(cstate)
    assert np.array_equal(xpo, xco)

    xpz = migo.features.zeros(pstate)
    xcz = migo.features.zeros(cstate)
    assert np.array_equal(xpz, xcz)

    for i in range(pstate.max_history_n):
        xp = migo.features.board_i(pstate, i)
        xc = migo.features.board_i(cstate, i)
        assert np.array_equal(xp, xc)

    xph = migo.features.history_n(pstate, n)
    xch = migo.features.history_n(cstate, n)
    assert xph.shape == xch.shape
    assert np.array_equal(xph, xch)
    xph = migo.features.migo.history_n(pstate, n)
    assert np.array_equal(xph, xch)


def test_consistent():
    pstate = migo.State(6)
    cstate = cygo.State(6)
    n = 3
    check_consistency(pstate, cstate, n)

    pstate.make_move((1, 2))
    cstate.make_move((1, 2))
    assert pstate.current_player == migo.Color.WHITE
    assert cstate.current_player == cygo.Color.WHITE

    check_consistency(pstate, cstate, n)

    pstate.make_move((4, 4))
    cstate.make_move((4, 4))
    assert pstate.current_player == migo.Color.BLACK
    assert cstate.current_player == cygo.Color.BLACK

    check_consistency(pstate, cstate, n)

    pstate.make_move(migo.misc.Pass)  # None
    cstate.make_move(migo.misc.Pass)  # None
    assert pstate.current_player == migo.Color.WHITE
    assert cstate.current_player == cygo.Color.WHITE

    pstate.make_move((3, 4))
    cstate.make_move((3, 4))
    assert pstate.current_player == migo.Color.BLACK
    assert cstate.current_player == cygo.Color.BLACK

    check_consistency(pstate, cstate, n)


def test_consistent_pass2():
    s = migo.State(9)
    s.make_move((1, 1))
    s.make_move((1, 2))
    x = migo.features.history_n(s, 4)  # history: (1, 1), (1, 2)
    s.make_move(None)
    s.make_move(None)
    y = migo.features.history_n(s, 4)   # history: (1, 1), (1, 2), pass, pass
    z = migo.features.migo.history_n(s, 4)
    assert not np.array_equal(x, y)
    assert not np.array_equal(x, z)
    assert np.array_equal(y, z)

    cs = cygo.State(9)
    cs.make_move((1, 1))
    cs.make_move((1, 2))
    cx = migo.features.history_n(cs, 4)  # history: (1, 1), (1, 2)
    cs.make_move(None)
    cs.make_move(None)
    cy = migo.features.history_n(cs, 4)   # history: (1, 1), (1, 2), pass, pass

    assert np.array_equal(x, cx)
    assert np.array_equal(y, cy)
    assert np.array_equal(z, cy)
