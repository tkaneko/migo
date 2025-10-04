"""Dataset for training with toarch."""
from .record import load_sgf_games_in_folder, SimpleRecord
import cygo
import migo.features
import torch
import numpy as np


class SgfDataset(torch.utils.data.Dataset):
    """Dataset serving as py:class:`torch.utils.data.Dataset`.

    :param folder_path: folder or list of folders containing sgf games
    :param history_n: specify input feature channels as (history_n + 1)*2
    :param batch_with_collate: use collate_fn in torch.utils.data.DataLoader
    :param ignore_opening_moves: 0 to use all moves.

    .. note:: assuming same board size for all records
    """

    def __init__(self, games: list = [],
                 history_n: int = 0, batch_with_collate: bool = False,
                 ignore_opening_moves: int = 0):
        from .dataset_transform import flip_ident
        # modifiable properties
        self.history_n = history_n
        self.batch_with_collate = batch_with_collate
        self.transform = flip_ident
        # should be constant, need rebuild if want to modify
        self.ignore_opening_moves = ignore_opening_moves
        total = 0
        if games:
            self.board_size = games[0].board_size
            self.winner = np.zeros(len(games), dtype=np.int8)
            for i, game in enumerate(games):
                total += len(game.moves)
                self.winner[i] = game.winner
                assert game.board_size == self.board_size
        if total > 0:
            #: concatenated moves
            self.game_moves = np.zeros(total, dtype=np.int16)
            #: game_id -> total moves before the game
            self.move_offset = np.zeros(len(games) + 1, dtype=np.int32)
            #: game_id -> total data before the game
            # (equivalent to move_offset if ignore_opening_moves == 0)
            self.data_offset = np.zeros(len(games) + 1, dtype=np.int32)
            idx = 0
            data_idx = 0
            for i, game in enumerate(games):
                self.move_offset[i] = idx
                self.data_offset[i] = data_idx
                self.game_moves[idx:idx+len(game.moves)] = game.moves[:]
                idx += len(game.moves)
                data_idx += max(0, len(game.moves) - ignore_opening_moves)
            self.move_offset[len(games)] = idx
            self.data_offset[len(games)] = data_idx
            assert total == idx

    def nth_game(self, n) -> SimpleRecord:
        '''extract a game of given index

        caveats: score is simplified in {-1, 0, 1}, and komi is a constant,
        for now
        '''
        return SimpleRecord(
            board_size=self.board_size,
            komi=7.0,           # todo!
            score=self.winner[n],
            moves=self.moves_view(n),
        )

    def n_games(self) -> int:
        """Return number of games."""
        return len(self.winner)

    def input_channels(self) -> int:
        """Return number of input channels in network to train."""
        return (self.history_n + 1) * 2 + 1

    def to_game_move_pair(self, flat_idx) -> tuple[int, int]:
        """Return pair of game id and move id.

        note: O(log N)
        """
        gid = np.searchsorted(self.data_offset, flat_idx, 'right') - 1
        move_id = flat_idx - self.data_offset[gid] + self.ignore_opening_moves
        return gid, move_id

    def moves_view(self, game_id) -> np.ndarray:
        """Return all moves in a game."""
        l, r = self.move_offset[game_id], self.move_offset[game_id + 1]
        return self.game_moves[l: r]

    @property
    def total_moves(self):
        """Return number of all moves in all games."""
        return self.move_offset[-1]

    @property
    def total_effective_moves(self):
        """Return number of all moves in all games."""
        return self.data_offset[-1]

    def __len__(self):
        """Return number of moves considering ignore_opening_moves"""
        return self.total_effective_moves

    def player_sgn(self, move_id):
        sgn = 1 if move_id % 2 == 0 else -1
        return sgn

    def winner_sgn(self, game_id, move_id):
        """Return 1 (-1) if result is win (loss), or 0 for draw.

        Win or loss is with respect for player to move.
        """
        sgn = self.player_sgn(move_id)
        return self.winner[game_id] * sgn

    def __getitem__(self, flat_idx):
        """Return a tuple of board_feature, move label, and winner label."""
        if self.batch_with_collate:
            return flat_idx
        gid, move_id = self.to_game_move_pair(flat_idx)
        moves = self.moves_view(gid)
        state = cygo.State(self.board_size, max_history_n=self.history_n)
        cygo.apply_moves(state, moves[:move_id])
        xh = migo.features.history_n(state, self.history_n)
        xc = migo.features.color(state)
        x = np.vstack((xh, xc))
        y_move = np.array([moves[move_id]])
        y_winner = self.winner_sgn(gid, move_id)
        self.transform(self.board_size, x, y_move)
        return (torch.from_numpy(x),
                torch.from_numpy(y_move).long(),
                torch.Tensor([y_winner]))

    def _collate_naive(self, indices):
        """Pure Python implementation compatible with collate."""
        N = len(indices)
        channels_per_state = 2 * (self.history_n + 1) + 1
        x = np.zeros(
            (N, channels_per_state, self.board_size, self.board_size),
            dtype=float
        )
        move_labels = np.zeros(N)
        value_labels = np.zeros(N)
        for i, flat_idx in enumerate(indices):
            gid, move_id = self.to_game_move_pair(flat_idx)
            moves = self.moves_view(gid)
            state = cygo.State(self.board_size, max_history_n=self.history_n)
            cygo.apply_moves(state, moves[:move_id])
            xh = migo.features.history_n(state, self.history_n)
            xc = migo.features.color(state)
            x[i, :, :, :] = np.vstack((xh, xc))
            move_labels[i] = moves[move_id]
            value_labels[i] = self.winner_sgn(gid, move_id)
        self.transform(self.board_size, x, move_labels)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).long().unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                )

    def set_transform(self, seed: int | str):
        from .dataset_transform import transforms, transforms_dict
        if isinstance(seed, str):
            self.transform = transforms_dict[seed]
        else:
            seed %= len(transforms)
            self.transform = transforms[seed]

    def _collate_cxx(self, indices):
        """Efficient implementation thanks to C++."""
        x, move_labels, value_labels = cygo.features.collate(
            indices,
            self.history_n, self.board_size,
            self.move_offset, self.game_moves, self.winner,
            self.data_offset, self.ignore_opening_moves,
        )
        self.transform(self.board_size, x, move_labels)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                )

    def collate(self, indices):
        """Collate function for torch.utils.data.DataLoader."""
        # return self._collate_naive(indices)
        return self._collate_cxx(indices)

    def to_img(self, flat_idx):
        """Visualize specified data.

        Colors are relative to turn to move in feature planes,
        except for the leftmost figure showing current state.
        """
        import migo.drawing
        import matplotlib.pyplot as plt
        gid, move_id = self.to_game_move_pair(flat_idx)
        moves = self.moves_view(gid)
        state = cygo.State(self.board_size, max_history_n=self.history_n)
        cygo.apply_moves(state, moves[:move_id])
        x = migo.features.history_n(state, self.history_n)
        fig, axs = plt.subplots(1, 1+len(x), figsize=((1+len(x))*3.3, 3.7))
        cmove = cygo.Move.from_raw_value(moves[move_id],
                                         board_size=self.board_size)
        coord = (cmove.row, cmove.col)
        # draw main board
        migo.drawing.setup_board_ax(self.board_size, fig, axs[0])
        scale = migo.drawing.ax_scale(9, fig, axs[0])
        migo.drawing.place_stones(axs[0], state, scale=scale)
        migo.drawing.put_number(
            axs[0], self.board_size, 'k', coord, 'a',
            with_shade=True, scale=scale**0.5
        )
        wlmsg = {1: 'win', 0: 'draw', -1: 'loss'}
        winlabel = self.winner_sgn(gid, move_id)
        migo.drawing.put_number(
            axs[0], self.board_size, 'k', (3.5, -1),
            f'{state.current_player.name} to play ({wlmsg[winlabel]})',
            scale=scale**0.5
        )

        # draw feature planes
        for i, plane in enumerate(x):
            migo.drawing.setup_board_ax(self.board_size, fig, axs[i+1])
            color = cygo.Color.BLACK if i % 2 == 0 else cygo.Color.WHITE
            migo.drawing.draw_plane(axs[i+1], plane, color,
                                    scale=scale)
            migo.drawing.put_number(
              axs[i+1], self.board_size, 'k', (3, -1),
              f'feature plane {i}', scale=scale**0.5
            )
        return fig

    def npy_to_img(self, flat_idx):
        """Visualize specified feature data.

        .. warning:: current coordinate system has 90 degrees difference
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        x, *ys = self[flat_idx]
        ncols, nrows = len(x), 1
        fig = plt.figure(figsize=(ncols*2.5, nrows*2))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                         axes_pad=0.3, label_mode='all')
        for i, ax in enumerate(grid):
            ax.imshow(x[i], cmap='Oranges')
            ax.invert_yaxis()

    def save_to(self, path) -> None:
        """Save all data to npz for future use without parsing sgf again."""
        np.savez_compressed(
            path,
            board_size=self.board_size,
            history_n=int(self.history_n),
            winner=self.winner,
            game_moves=self.game_moves,
            game_index=self.move_offset,  # keeps old key for compatibility
            ignore_opening_moves=self.ignore_opening_moves,
            data_offset=self.data_offset,
            dataset_type=np.array('SgfDataset', dtype=np.bytes_),
        )

    @staticmethod
    def build_from_folder(folder_path: str | list[str], *, batch_with_collate):
        if isinstance(folder_path, str):
            folder_path = [folder_path]
        board_size = None
        games = []
        for folder in folder_path:
            loaded = load_sgf_games_in_folder(folder)
            if not loaded:
                continue
            games += loaded
            if not board_size:
                board_size = games[0].board_size
            elif board_size != loaded[0].board_size:
                raise RuntimeError(f'board size mismatch {board_size}'
                                   f' != {loaded[0].board_size}')
        return SgfDataset(games, batch_with_collate=batch_with_collate)

    @staticmethod
    def load_from_objs(objs, *, batch_with_collate):
        """Load ojbs loaded from npz.
        """
        ret = SgfDataset([], batch_with_collate=batch_with_collate)
        ret.board_size = int(objs['board_size'])
        ret.history_n = int(objs['history_n'])
        ret.winner = objs['winner']
        ret.game_moves = objs['game_moves']
        ret.move_offset = objs['game_index'].astype(np.int32)
        ret.ignore_opening_moves = objs.get('ignore_opening_moves', 0)
        if ret.ignore_opening_moves > 0:
            ret.data_offset = objs['data_offset'].astype(np.int32)
        else:
            ret.data_offset = ret.move_offset
        return ret

    @staticmethod
    def load_from(path, *, batch_with_collate):
        """Load npz file (saved by `save_to`) to construct `SgfDataset`.

        usage: `dataset = migo.SgfDataset.load_from('path-to-npz')`
        """
        objs = np.load(path)
        return SgfDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )

    def summary(self) -> dict:
        n_games = self.n_games()
        return {
            'dbtype': 'SgfDataset',
            'games': n_games,
            'moves-per-game': self.move_offset[-1]/n_games,
            'data-per-game': self.data_offset[-1]/n_games,
            'average-value': np.average(self.winner),
            'ignore-opening': self.ignore_opening_moves,
            'input-channels': self.input_channels(),
        }

    def append(self, other) -> None:
        '''merge another SgfDataset at the end
        '''
        if (
                self.history_n != other.history_n
                or self.board_size != other.board_size
                or self.ignore_opening_moves != other.ignore_opening_moves
        ):
            raise ValueError('type mismatch')
        self.winner = np.append(self.winner, other.winner)
        self.game_moves = np.append(self.game_moves, other.game_moves)

        offset = self.move_offset[-1]
        self.move_offset = np.append(
            self.move_offset[:-1],
            other.move_offset + offset
        )
        offset_data = self.data_offset[-1]
        self.data_offset = np.append(
            self.data_offset[:-1],
            other.data_offset + offset_data
        )
        assert self.move_offset[-1] == self.total_moves
        assert len(self.winner) + 1 == len(self.move_offset)
        assert len(self.game_moves) == self.total_moves


class ZoneDataset(torch.utils.data.Dataset):
    '''extention of SgfDataset with zones (to be replaced by ExtendedDataset)

    - owns (not inherits) SgfDataset
    - maintains each player's zone (area of interests) and the
      corresponding score for each game
    '''
    def __init__(self, sgfdataset: SgfDataset, zones, zone_scores):
        self.sgf_dataset = sgfdataset
        self.zones = zones
        self.zone_scores = zone_scores
        size = sgfdataset.board_size
        assert zones.shape == (sgfdataset.n_games(), 2, size, size)
        assert len(zone_scores) == sgfdataset.n_games()

    def __len__(self):
        """Return number of all moves in all games."""
        return len(self.sgf_dataset)

    def __getitem__(self, flat_idx):
        '''return id itself assuming collate function'''
        return flat_idx

    def n_games(self) -> int:
        """Return number of games."""
        return self.sgf_dataset.n_games()

    def input_channels(self) -> int:
        """Return number of input channels in network to train."""
        return self.sgf_dataset.input_channels() + 1

    @property
    def board_size(self) -> int:
        return self.sgf_dataset.board_size

    def set_transform(self, seed: int | str):
        self.sgf_dataset.set_transform(seed)

    def collate(self, indices):
        """Collate function for torch.utils.data.DataLoader."""
        # return self._collate_naive(indices)
        return self._collate_cxx(indices)

    def _collate_cxx(self, indices):
        """Efficient implementation thanks to C++."""
        base = self.sgf_dataset
        ret = cygo.features.collatez(
            # original parameters
            indices,
            base.history_n, base.board_size,
            base.move_offset, base.game_moves, base.winner,
            base.data_offset, base.ignore_opening_moves,
            # extended ones
            self.zones.reshape(-1), self.zone_scores.reshape(-1),
        )
        x, move_labels, value_labels, zone_labels = ret
        base.transform(base.board_size, x, move_labels)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                torch.from_numpy(zone_labels).unsqueeze(-1),
                )

    def _collate_naive(self, indices):
        """Pure Python implementation compatible with collate."""
        base = self.sgf_dataset
        board_size = base.board_size
        history_n = base.history_n
        N = len(indices)
        channels_per_state = 2 * (history_n + 1) + 1 + 1
        x = np.zeros(
            (N, channels_per_state, board_size, board_size),
            dtype=float
        )
        move_labels = np.zeros(N)
        value_labels = np.zeros(N)
        zone_scores = np.zeros(N)
        for i, flat_idx in enumerate(indices):
            gid, move_id = base.to_game_move_pair(flat_idx)
            moves = base.moves_view(gid)
            state = cygo.State(board_size, max_history_n=history_n)
            cygo.apply_moves(state, moves[:move_id])
            xh = migo.features.history_n(state, history_n)
            xc = migo.features.color(state)
            xz = self.zones[gid][move_id % 2]
            x[i, :, :, :] = np.vstack((xh, xc, xz[np.newaxis, :, :]))  # new!
            move_labels[i] = moves[move_id]
            value_labels[i] = base.winner_sgn(gid, move_id)
            sgn = base.player_sgn(move_id)
            zone_scores[i] = self.zone_scores[gid] * sgn
        base.transform(base.board_size, x, move_labels)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).long().unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                torch.from_numpy(zone_scores).unsqueeze(-1),
                )

    def append(self, other) -> None:
        '''merge another ZoneDataset at the end
        '''
        self.sgf_dataset.append(other.sgf_dataset)
        self.zones = np.vstack([self.zones, other.zones])
        self.zone_scores = np.append(self.zone_scores, other.zone_scores)

    def save_to(self, path) -> None:
        """Save all data to npz for future use without parsing sgf again."""
        base = self.sgf_dataset
        np.savez_compressed(
            # for SgfDataset
            path,
            board_size=base.board_size,
            history_n=int(base.history_n),
            winner=base.winner,
            game_moves=base.game_moves,
            game_index=base.move_offset,
            ignore_opening_moves=base.ignore_opening_moves,
            data_offset=base.data_offset,
            dataset_type=np.array('ZoneDataset', dtype=np.bytes_),
            # ZoneDataset
            zones=self.zones,
            zone_scores=self.zone_scores,
        )

    @staticmethod
    def load_from_objs(objs, *, batch_with_collate):
        """Load npz file (saved by `save_to`) to construct `ZoneDataset`.

        usage: `dataset = migo.ZoneDataset.load_from('path-to-npz')`
        """
        base = SgfDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )
        return ZoneDataset(
            base, objs['zones'], objs['zone_scores']
        )

    @staticmethod
    def load_from(path, *, batch_with_collate):
        """Load npz file (saved by `save_to`) to construct `ZoneDataset`.

        usage: `dataset = migo.ZoneDataset.load_from('path-to-npz')`
        """
        objs = np.load(path)
        return ZoneDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )

    def summary(self) -> dict:
        ret = self.sgfdataset.summary()
        ret['dbtype'] = 'ZoneDataset'
        ret['input-channels'] = self.input_channels()
        return ret


class ExtendedDataset(torch.utils.data.Dataset):
    '''alternative extention of SgfDataset with auxiliary data

    - owns (not inherits) SgfDataset
    - maintains each player's zone (area of interests) and the
      corresponding score for each game
    - configurable color of interests
    - sample by game id
    '''
    def __init__(
            self, sgfdataset: SgfDataset, aux_input_planes,
            aux_values, aux_oplanes=None
    ):
        '''
        :param aux_input_plane: added to input features, e.g., zone
        :param aux_value: auxiliary value for each game
        '''
        self.sgf_dataset = sgfdataset
        self.aux_input_planes = np.array(aux_input_planes, dtype=np.int8)
        self.aux_values = np.array(aux_values, dtype=np.float32)
        if (abs_max := max(abs(aux_values[0]).max(),
                           abs(aux_values[1]).max())) > 1:
            raise ValueError(f'aux values not in range [-1, 1] {abs_max}')
        size = sgfdataset.board_size
        assert self.aux_input_planes.shape == (2, size, size)
        assert aux_values.shape == (2, sgfdataset.n_games())
        self.enabled_colors = [True, True]
        self.aux_oplanes = aux_oplanes
        if aux_oplanes is None:
            self.aux_oplanes = cygo.features.make_territory(
                size,
                sgfdataset.move_offset,
                sgfdataset.game_moves,
            )

    def __len__(self):
        """Return number of all moves in all games."""
        # n_colors = sum(self.enabled_colors)
        # scale = n_colors // 2
        return len(self.sgf_dataset)              # * scale

    def __getitem__(self, flat_idx):
        '''return id itself assuming collate function'''
        return flat_idx

    def n_games(self) -> int:
        """Return number of games."""
        return self.sgf_dataset.n_games()

    def input_channels(self) -> int:
        """Return number of input channels in network to train."""
        return self.sgf_dataset.input_channels() + 1

    @property
    def board_size(self) -> int:
        return self.sgf_dataset.board_size

    def enable_color(self, black, white):
        self.enabled_colors = [black, white]

    def set_transform(self, seed: int | str):
        self.sgf_dataset.set_transform(seed)

    def collate(self, indices):
        """Collate function for torch.utils.data.DataLoader."""
        # return self._collate_naive(indices)
        return self._collate_cxx(indices)

    def _collate_cxx(self, indices):
        """Efficient implementation thanks to C++."""
        base = self.sgf_dataset
        ret = cygo.features.collate_ext(
            # original parameters
            indices,
            base.history_n, base.board_size,
            base.move_offset, base.game_moves, base.winner,
            base.data_offset, base.ignore_opening_moves,
            # extended ones
            self.enabled_colors,
            self.aux_input_planes.reshape(-1),
            self.aux_oplanes.reshape(-1),
            self.aux_values.reshape(-1)
        )
        x, move_labels, value_labels, aux_oplanes, aux_labels = ret
        base.transform(base.board_size, x, move_labels, aux_oplanes)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                torch.from_numpy(aux_oplanes),
                torch.from_numpy(aux_labels).unsqueeze(-1),
                )

    def _collate_naive(self, indices):
        """Pure Python implementation compatible with collate."""
        base = self.sgf_dataset
        board_size = base.board_size
        history_n = base.history_n
        N = len(indices)
        channels_per_state = 2 * (history_n + 1) + 1 + 1
        x = np.zeros(
            (N, channels_per_state, board_size, board_size),
            dtype=float
        )
        move_labels = np.zeros(N, dtype=np.int32)
        value_labels = np.zeros(N, dtype=np.int8)
        aux_values = np.zeros(N, dtype=np.float32)
        aux_oplanes = np.zeros((N, (board_size ** 2 + 1)), dtype=np.float32)
        for i, flat_idx in enumerate(indices):
            gid, move_id = base.to_game_move_pair(flat_idx)
            if not self.enabled_colors[move_id % 2]:
                move_id = move_id - 1 if move_id > 0 else move_id + 1
            moves = base.moves_view(gid)
            state = cygo.State(board_size, max_history_n=history_n)
            cygo.apply_moves(state, moves[:move_id])
            xh = migo.features.history_n(state, history_n)
            xc = migo.features.color(state)
            xz = self.aux_input_planes[move_id % 2]
            x[i, :, :, :] = np.vstack((xh, xc, xz[np.newaxis, :, :]))  # new!
            move_labels[i] = moves[move_id]
            value_labels[i] = base.winner_sgn(gid, move_id)
            aux_values[i] = self.aux_values[move_id % 2][gid]
            cygo.apply_moves(state, moves[move_id:])
            tt = state.tromp_taylor_fill().reshape(-1)
            aux_oplanes[i, :board_size**2] = tt * base.player_sgn(move_id)

        base.transform(board_size, x, move_labels, aux_oplanes)
        return (torch.from_numpy(x),
                torch.from_numpy(move_labels).long().unsqueeze(-1),
                torch.from_numpy(value_labels).unsqueeze(-1),
                torch.from_numpy(aux_oplanes),
                torch.from_numpy(aux_values).unsqueeze(-1),
                )

    def append(self, other) -> None:
        '''merge another Extended at the end
        '''
        self.sgf_dataset.append(other.sgf_dataset)
        if not np.array_equal(self.aux_input_planes, other.aux_input_planes):
            raise ValueError('incompatible aux_input_planes')
        self.aux_values = np.concatenate(
            (self.aux_values, other.aux_values), axis=1
        )
        self.aux_oplanes = np.append(self.aux_oplanes, other.aux_oplanes)

    def save_to(self, path) -> None:
        """Save all data to npz for future use without parsing sgf again."""
        base = self.sgf_dataset
        np.savez_compressed(
            # for SgfDataset
            path,
            board_size=base.board_size,
            history_n=int(base.history_n),
            winner=base.winner,
            game_moves=base.game_moves,
            game_index=base.move_offset,
            ignore_opening_moves=base.ignore_opening_moves,
            data_offset=base.data_offset,
            dataset_type=np.array('ExtendedDataset', dtype=np.bytes_),
            # ExtendedDataset
            aux_input_planes=self.aux_input_planes,
            aux_oplanes=self.aux_oplanes,
            aux_values=self.aux_values,
        )

    @staticmethod
    def load_from_objs(objs, *, batch_with_collate):
        """Load npz objs.
        """
        base = SgfDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )
        return ExtendedDataset(
            base, objs['aux_input_planes'],
            objs['aux_values'],
            objs.get('aux_oplanes', None)
        )

    @staticmethod
    def load_from(path, *, batch_with_collate):
        """Load npz file (saved by `save_to`) to construct `ExtendedDataset`.

        usage: `dataset = migo.ExtendedDataset.load_from('path-to-npz')`
        """
        objs = np.load(path)
        return ExtendedDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )

    @staticmethod
    def build_from(base: SgfDataset,
                   zones, *, batch_with_collate):
        board_size = base.board_size
        if zones.shape != (2, board_size, board_size):
            raise ValueError(f'dimension mismatch {zones.shape}')
        territory = cygo.features.make_territory(
            board_size,
            base.move_offset,
            base.game_moves,
        ).reshape(-1, 82)
        zone_area = [max(1, z.sum()) for z in zones]
        zone_score = np.zeros((2, base.n_games()), dtype=np.float32)
        zone_score[0] = np.dot(
            np.maximum(territory[:, :81], 0),
            zones[0].reshape(-1).astype(float)
        ) / zone_area[0]
        zone_score[1] = -np.dot(  # as white stone be -1 in territory
            np.minimum(territory[:, :81], 0),
            zones[1].reshape(-1).astype(float)
        ) / zone_area[1]
        zone_score = zone_score * 2 - 1  # [0,1] -> [-1, 1]
        return ExtendedDataset(base, zones, zone_score, territory)

    def summary(self) -> dict:
        ret = self.sgf_dataset.summary()
        ret['dbtype'] = 'ExtendedDataset'
        ret['input-channels'] = self.input_channels()
        ret['aux-input-black-hot'] = self.aux_input_planes[0].sum()
        ret['aux-input-white-hot'] = self.aux_input_planes[1].sum()
        ret['mean-aux-value-black'] = self.aux_values[0].mean()
        ret['mean-aux-value-white'] = self.aux_values[1].mean()
        ret['var-aux-value-black'] = self.aux_values[0].var()
        ret['var-aux-value-white'] = self.aux_values[1].var()
        return ret


def load_dataset(path_or_objs, *, batch_with_collate):
    objs = path_or_objs
    if not isinstance(path_or_objs, dict):
        objs = np.load(path_or_objs)
    if 'aux_input_planes' in objs:
        return ExtendedDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )
    if 'zones' in objs:
        return ZoneDataset.load_from_objs(
            objs, batch_with_collate=batch_with_collate
        )
    return SgfDataset.load_from_objs(
        objs, batch_with_collate=batch_with_collate
    )
