import migo
import torch
import numpy as np
import numpy.testing as npt
import pytest

history_n = 4
short_sgf_str = [
    '(;FF[4]GM[1]SZ[9]KM[7]RE[Draw];B[ee];W[he];B[tt];W[tt])',
    '(;FF[4]GM[1]SZ[9]KM[7]RE[B+R];B[cb])',
    '(;FF[4]GM[1]SZ[9]KM[7]RE[W+R];B[hf];W[cg])',
]
sgf_str = [
    '(;GM[1]SZ[9]KM[7.0]RE[B+2.0]RU[Chinese];'
    'B[ee];W[ec];B[cd];W[ce];B[dd];W[ed];B[de];W[cf];B[ge];W[gc];B[dg];'
    'W[df];B[ef];W[cg];B[eh];W[dh];B[eg];W[di];B[hd];W[hc];B[db];W[gd];'
    'B[he];W[eb];B[be];W[bf];B[bh];W[ch];B[ae];W[bi];B[ei];W[dc];B[cc];'
    'W[cb];B[bb];W[da];B[ic];W[ib];B[id];W[ah];B[hb];W[gb];B[ia];W[ba];'
    'B[bc];W[fe];B[ff];W[fd];B[af];W[ab];B[ib];W[ca];B[ac];W[ga];B[aa];'
    'W[gf];B[hf];W[gg];B[hg];W[gh];B[hh];W[hi];B[gi];W[fi];B[ab];W[ag];'
    'B[ih];W[ha];B[fh];W[bg];B[gi];W[bd];B[ad];W[fi];B[ea];W[fa];B[gi];'
    'W[if];B[fg];W[fb];B[ig];W[gh];B[ii];W[gf];B[gg];W[tt];B[ie];W[tt];'
    'B[tt])',
    '(;GM[1]SZ[9]KM[7.0]GN[R=2,,0]RE[Draw]RU[Chinese];'
    'B[dd];W[ef];B[fe];W[gg];B[cf];W[fc];B[ec];W[eb];B[fb];W[gb];B[db];'
    'W[fa];B[fd];W[gd];B[dg];W[ge];B[ff];W[fg];B[eg];W[da];B[ca];W[gf];'
    'B[ea];W[ed];B[ee];W[da];B[cb];W[eh];B[dh];W[fh];B[ea];W[de];B[df];'
    'W[da];B[ei];W[fi];B[ea];W[fb];B[di];W[da];B[hc];W[ea];B[gc];W[hb];'
    'B[ib];W[hd];B[ic];W[id];B[ha];W[ia];B[ib];W[ic];B[ce];W[ga];B[gc];'
    'W[ia];B[cc];W[hf];B[bh];W[ib];B[hh];W[hi];B[gi];W[gh];B[ii];W[ih];'
    'B[ig];W[hg];B[ba];W[bc];B[bd];W[bb];B[ac];W[ab];B[aa];W[bi];B[ci];'
    'W[ah];B[ai];W[bg];B[ag];W[af];B[ah];W[ch];B[cg];W[bf];B[be];W[ae];'
    'B[ad];W[ii];B[bi];W[gi];B[ef];W[hc];B[if];W[ie];B[ig];W[if];B[bg];'
    'W[ig];B[cd];W[bf];B[af];W[bc];B[bb];W[tt];B[ed];W[tt];B[ab];W[tt];'
    'B[ae];W[tt];B[tt])',
]


def make_dataset(*, batch_with_collate: bool = False,
                 ignore_opening_moves: int = 0) -> migo.SgfDataset:
    games = [migo.record.parse_sgf_game(_) for _ in sgf_str]
    return migo.SgfDataset(
        games=games, history_n=history_n,
        batch_with_collate=batch_with_collate,
        ignore_opening_moves=ignore_opening_moves,
    )


def test_index():
    dataset = make_dataset()
    assert len(dataset) == 202
    assert dataset.to_game_move_pair(0) == (0, 0)
    assert dataset.to_game_move_pair(88) == (0, 88)
    assert dataset.to_game_move_pair(89) == (1, 0)


def test_opening():
    dataset = make_dataset(ignore_opening_moves=1)
    assert len(dataset) == 200
    assert dataset.to_game_move_pair(0) == (0, 1)
    assert dataset.to_game_move_pair(87) == (0, 88)
    assert dataset.to_game_move_pair(88) == (1, 1)

    dataset_org = make_dataset()
    x, move, value = dataset[0]
    x0, move0, value0 = dataset_org[1]
    assert move == move0
    assert value == value0
    assert torch.equal(x, x0)

    x, move, value = dataset._collate_cxx([0])
    x0, move0, value0 = dataset_org._collate_cxx([1])
    assert torch.equal(move, move0)
    assert torch.equal(value, value0)
    assert torch.equal(x, x0)

    dataset10 = make_dataset(ignore_opening_moves=10)
    assert len(dataset10) == 182
    assert dataset10.to_game_move_pair(0) == (0, 10)
    assert dataset10.to_game_move_pair(78) == (0, 88)
    assert dataset10.to_game_move_pair(79) == (1, 10)

    x, move, value = dataset10[0]
    x0, move0, value0 = dataset_org[10]
    assert move == move0
    assert value == value0
    assert torch.equal(x, x0)

    x, move, value = dataset10._collate_cxx([3])
    x0, move0, value0 = dataset_org._collate_cxx([13])
    assert torch.equal(move, move0)
    assert torch.equal(value, value0)
    assert torch.equal(x, x0)


def test_sign():
    dataset = make_dataset()
    N = len(dataset)
    # first game +B
    x, move, value = dataset[0]
    assert value == 1
    x, move, value = dataset[1]
    assert value == -1
    x, move, value = dataset[2]
    assert value == 1
    # second game draw
    x, move, value = dataset[N-1]
    assert value == 0
    x, move, value = dataset[N-2]
    assert value == 0


def test_dataset():
    batch_size = 32
    dataset = make_dataset()

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    batch = next(iter(trainloader))
    assert batch is not None

    dataset2 = make_dataset(batch_with_collate=True)

    assert np.array_equal(dataset.winner, dataset2.winner)
    trainloader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda indices: dataset2._collate_naive(indices),
    )
    batch2 = next(iter(trainloader2))
    assert batch2 is not None

    trainloader3 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda indices: dataset2._collate_cxx(indices),
    )
    batch3 = next(iter(trainloader3))
    assert batch3 is not None

    assert torch.equal(batch[1], batch2[1].long())  # move label
    assert torch.equal(batch[2], batch2[2])  # value label
    assert batch[0].shape == batch2[0].shape
    assert torch.equal(batch[0], batch2[0])  # input feature

    assert torch.equal(batch[1], batch3[1].long())  # move label
    assert torch.equal(batch[2], batch3[2])  # value label
    assert batch[0].shape == batch3[0].shape
    assert torch.equal(batch[0], batch3[0])  # input feature


def test_collate():
    dataset = make_dataset()
    full = list(range(min(96, len(dataset))))
    x, move, value = dataset._collate_naive(full)
    x2, move2, value2 = dataset._collate_cxx(full)
    assert np.array_equal(value.numpy(), value2.numpy())
    assert torch.equal(move, move2)
    assert torch.equal(x, x2)


def test_append():
    games = [migo.record.parse_sgf_game(_) for _ in short_sgf_str]
    all = migo.SgfDataset(games=games, history_n=1)
    assert len(all) == 7

    other = migo.SgfDataset(games=[games[0]], history_n=1)
    for i in range(1, len(games)):
        other.append(migo.SgfDataset(
            games=[games[i]], history_n=1
        ))
    assert all.n_games() == other.n_games()
    assert len(all) == len(other)
    assert len(all) > 0

    indices = np.array(list(range(len(all))), dtype=np.int32)
    d1, m1, w1 = all._collate_cxx(indices)
    d2, m2, w2 = other._collate_cxx(indices)

    npt.assert_equal(d1.numpy(), d2.numpy())
    npt.assert_equal(m1.numpy(), m2.numpy())
    npt.assert_equal(w1.numpy(), w2.numpy())


def test_append_with_opening():
    ign = 1
    games = [migo.record.parse_sgf_game(_) for _ in short_sgf_str]
    all = migo.SgfDataset(
        games=games, history_n=1,
        ignore_opening_moves=ign
    )
    assert len(all) == 4

    other = migo.SgfDataset(
        games=[games[0]], history_n=1,
        ignore_opening_moves=ign
    )
    for i in range(1, len(games)):
        other.append(migo.SgfDataset(
            games=[games[i]], history_n=1,
            ignore_opening_moves=ign
        ))
    assert all.n_games() == other.n_games()
    assert len(all) == len(other)
    assert len(all) > 0

    indices = np.array(list(range(len(all))), dtype=np.int32)
    d1, m1, w1 = all._collate_cxx(indices)
    d2, m2, w2 = other._collate_cxx(indices)

    npt.assert_equal(d1.numpy(), d2.numpy())
    npt.assert_equal(m1.numpy(), m2.numpy())
    npt.assert_equal(w1.numpy(), w2.numpy())


def test_zone_dataset():
    base_dataset = make_dataset()

    zones = np.zeros((2, 2, 9, 9), dtype=np.int8)
    zones[0, 0, 6, 7] = 1
    zones[0, 1, 3, 3] = 1
    zones[1, 0, 1, 2] = 1
    zones[1, 1, 5, 4] = 1
    zone_scores = np.array([0.3, 0.7], dtype=np.float32)

    zone_dataset = migo.ZoneDataset(base_dataset, zones, zone_scores)
    channels = (history_n+1)*2 + 2

    # first, second, and third moves
    idx = [0, 1, 2]
    x, move, value, zscore = zone_dataset._collate_naive(idx)
    assert x.shape == (len(idx), channels, 9, 9)
    for bi, fi in enumerate(idx):
        assert np.array_equal(x[bi, channels-1, :, :].numpy(),
                              zones[0][fi % 2])
        sgn = (-1)**(fi % 2)
        assert pytest.approx(zscore[bi].item()) == sgn * zone_scores[0]

    x2, move2, value2, zscore2 = zone_dataset._collate_cxx(idx)
    for bi, fi in enumerate(idx):
        sgn = (-1)**(fi % 2)
        assert pytest.approx(zscore2[bi].item()) == sgn * zone_scores[0]
        assert np.array_equal(x2[bi, channels-1, :, :].numpy(),
                              zones[0][fi % 2])
    assert torch.equal(x, x2)
    assert torch.equal(move, move2)
    assert torch.equal(value, value2)
    assert torch.equal(zscore, zscore2)

    # between games
    x, move, value, zscore = zone_dataset._collate_naive([88, 89])
    assert np.array_equal(x[0, channels-1, :, :].numpy(), zones[0][0])
    assert np.array_equal(x[1, channels-1, :, :].numpy(), zones[1][0])
    assert zscore[0] == zone_scores[0]
    assert zscore[1] == zone_scores[1]
    x, move, value, zscore = zone_dataset._collate_cxx([88, 89])
    assert np.array_equal(x[0, channels-1, :, :].numpy(), zones[0][0])
    assert np.array_equal(x[1, channels-1, :, :].numpy(), zones[1][0])
    assert zscore[0] == zone_scores[0]
    assert zscore[1] == zone_scores[1]

    # full compatibility
    full = [89]  # list(range(len(base_dataset)))
    x, move, value, zscore = zone_dataset._collate_naive(full)
    x2, move2, value2, zscore2 = zone_dataset._collate_cxx(full)

    assert torch.equal(move, move2)
    assert torch.equal(value, value2)
    assert torch.equal(zscore, zscore2)
    assert torch.equal(x, x2)


def test_append_zone():
    zones = np.zeros((len(short_sgf_str), 2, 9, 9), dtype=np.int8)
    for i in range(len(short_sgf_str)):
        zones[i, 0, i, i+1] = 1
        zones[i, 1, i+2, i+3] = 1
    zone_scores = np.arange(len(short_sgf_str), dtype=np.float32)

    games = [migo.record.parse_sgf_game(_) for _ in short_sgf_str]
    all_base = migo.SgfDataset(games=games, history_n=1)
    all = migo.ZoneDataset(all_base, zones, zone_scores)

    other_base = migo.SgfDataset(games=[games[0]], history_n=1)
    other = migo.ZoneDataset(
        other_base, np.array([zones[0]]), np.array([zone_scores[0]])
    )
    for i in range(1, len(games)):
        base = migo.SgfDataset(
            games=[games[i]], history_n=1
        )
        new_data = migo.ZoneDataset(
            base, np.array([zones[i]]), np.array([zone_scores[i]])
        )
        other.append(new_data)
    assert len(all) == len(other)
    assert len(all) > 0

    indices = np.array(list(range(len(all))), dtype=np.int32)
    d1, m1, w1, s1 = all._collate_cxx(indices)
    d2, m2, w2, s2 = other._collate_cxx(indices)

    npt.assert_equal(d1.numpy(), d2.numpy())
    npt.assert_equal(m1.numpy(), m2.numpy())
    npt.assert_equal(w1.numpy(), w2.numpy())
    npt.assert_equal(s1.numpy(), s2.numpy())


def test_collate_extended():
    base = make_dataset()
    aux_zones = np.zeros((2, 9, 9), dtype=np.int8)
    aux_zones[0, 1, 2] = 1
    aux_zones[1, 5, 4] = 1
    aux_scores = np.arange(2*base.n_games(), dtype=np.float32).reshape(2, -1) / (2*base.n_games())
    dataset = migo.ExtendedDataset(base, aux_zones, aux_scores)
    full = list(range(min(96, len(dataset))))

    # both colors
    x, move, value, aux_plane, aux_value = dataset._collate_naive(full)
    x2, move2, value2, aux_plane2, aux_value2 = dataset._collate_cxx(full)
    assert np.array_equal(value.numpy(), value2.numpy())
    assert torch.equal(move, move2)
    assert torch.equal(x, x2)
    assert torch.equal(aux_value, aux_value2)
    assert aux_plane.shape == aux_plane2.shape
    assert torch.equal(aux_plane, aux_plane2)

    # black only
    dataset.enable_color(True, False)
    x, move, value, aux_plane, aux_value = dataset._collate_naive(full)
    x2, move2, value2, aux_plane2, aux_value2 = dataset._collate_cxx(full)
    assert np.array_equal(value.numpy(), value2.numpy())
    assert torch.equal(move, move2)
    assert torch.equal(x, x2)
    assert torch.equal(aux_value, aux_value2)
    assert torch.equal(aux_plane, aux_plane2)

    # white only
    dataset.enable_color(False, True)
    x, move, value, aux_plane, aux_value = dataset._collate_naive(full)
    x2, move2, value2, aux_plane2, aux_value2 = dataset._collate_cxx(full)
    assert np.array_equal(value.numpy(), value2.numpy())
    assert torch.equal(move, move2)
    assert torch.equal(x, x2)
    assert torch.equal(aux_value, aux_value2)
    assert torch.equal(aux_plane, aux_plane2)


def test_append_extended():
    games = [migo.record.parse_sgf_game(_) for _ in short_sgf_str]
    all_base = migo.SgfDataset(games=games, history_n=1)

    aux_zones = np.zeros((2, 9, 9), dtype=np.int8)
    aux_zones[0, 1, 2] = 1
    aux_zones[1, 5, 4] = 1
    aux_scores = np.arange(
        2*all_base.n_games(), dtype=np.float32
    ).reshape(2, -1) / (2*all_base.n_games())
    all = migo.ExtendedDataset(all_base, aux_zones, aux_scores)

    other_base = migo.SgfDataset(games=[games[0]], history_n=1)
    print(f'{aux_scores.shape=} {aux_scores[:, 0, np.newaxis].shape=}')
    other = migo.ExtendedDataset(
        other_base, aux_zones, aux_scores[:, 0, np.newaxis]
    )
    for i in range(1, len(games)):
        base = migo.SgfDataset(
            games=[games[i]], history_n=1
        )
        new_data = migo.ExtendedDataset(
            base, aux_zones, aux_scores[:, i, np.newaxis]
        )
        other.append(new_data)
    assert len(all) == len(other)
    assert len(all) > 0

    indices = np.array(list(range(len(all))), dtype=np.int32)
    d1, m1, w1, ap1, s1 = all._collate_cxx(indices)
    d2, m2, w2, ap2, s2 = other._collate_cxx(indices)

    npt.assert_equal(d1.numpy(), d2.numpy())
    npt.assert_equal(m1.numpy(), m2.numpy())
    npt.assert_equal(w1.numpy(), w2.numpy())
    npt.assert_equal(ap1.numpy(), ap2.numpy())
    npt.assert_equal(s1.numpy(), s2.numpy())


def test_flip_lr():
    from migo.dataset_transform import flip_lr
    offset = 100
    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    id = np.array([list(range(9))+[-1]])
    flip_lr(3, bd, id)
    assert np.array_equal(
        bd,
        np.array([[[2, 1, 0], [5, 4, 3], [8, 7, 6]]])+offset
    )
    id2 = np.array([[2, 1, 0, 5, 4, 3, 8, 7, 6, 9]])
    assert np.array_equal(id, id2)

    bd = np.arange(9).reshape(1, 3, 3)
    aux_bd = np.arange(offset, offset+10).reshape(1, 10)
    id = np.array([list(range(9))+[-1]])
    flip_lr(3, bd, id, aux_bd)
    assert np.array_equal(bd, [[[2, 1, 0], [5, 4, 3], [8, 7, 6]]])
    assert np.array_equal(id, id2)
    assert np.array_equal(aux_bd, id2+100)


def test_flip_udlr():
    from migo.dataset_transform import flip_udlr
    offset = 100
    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    id = np.array([list(range(9))+[-1]])
    flip_udlr(3, bd, id)
    rev = list(reversed(range(9)))
    assert np.array_equal(
        bd,
        np.array(rev).reshape(1, 3, 3)+offset
    )
    id2 = np.array([rev + [9]])
    assert np.array_equal(id, id2)

    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    aux_bd = np.arange(offset, offset+10).reshape(1, 10)
    id = np.array([list(range(9))+[-1]])
    flip_udlr(3, bd, id, aux_bd)

    assert np.array_equal(
        bd,
        np.array(rev).reshape(1, 3, 3)+offset
    )
    assert np.array_equal(id, id2)
    aux_bd2 = np.array([rev + [9]]) + offset
    assert np.array_equal(
        aux_bd,
        aux_bd2
    )


def test_flip_ud():
    from migo.dataset_transform import flip_ud
    offset = 100
    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    id = np.array([list(range(9))+[-1]])
    flip_ud(3, bd, id)
    bd_v = [6, 7, 8, 3, 4, 5, 0, 1, 2]
    assert np.array_equal(
        bd,
        np.array(bd_v).reshape(1, 3, 3)+offset
    )
    id2 = np.array([bd_v + [9]])
    assert np.array_equal(id, id2)

    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    aux_bd = np.arange(offset, offset+10).reshape(1, 10)
    id = np.array([list(range(9))+[-1]])
    flip_ud(3, bd, id, aux_bd)

    assert np.array_equal(
        bd,
        np.array(bd_v).reshape(1, 3, 3)+offset
    )
    assert np.array_equal(id, id2)
    aux_bd2 = np.array([bd_v + [9]]) + offset
    assert np.array_equal(
        aux_bd,
        aux_bd2
    )


def test_rot90():
    from migo.dataset_transform import rot90
    offset = 100
    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    id = np.array([list(range(9))+[-1]])
    rot90(3, bd, id)
    bd90 = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    bdn90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    assert np.array_equal(
        bd,
        np.array(bd90).reshape(1, 3, 3)+offset
    )
    id2 = np.array([bdn90 + [9]])
    assert np.array_equal(id, id2)

    bd = np.arange(offset, offset+9).reshape(1, 3, 3)
    aux_bd = np.arange(offset, offset+10).reshape(1, 10)
    id = np.array([list(range(9))+[-1]])
    rot90(3, bd, id, aux_bd)

    assert np.array_equal(
        bd,
        np.array(bd90).reshape(1, 3, 3)+offset
    )
    assert np.array_equal(id, id2)
    aux_bd2 = np.array([bd90 + [9]]) + offset
    assert np.array_equal(
        aux_bd,
        aux_bd2
    )
