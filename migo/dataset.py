from .record import load_sgf_games_in_folder
import cygo
import migo.features
import torch
import numpy as np


class SgfDataset(torch.utils.data.Dataset):
    """dataset serving as py:class:`torch.utils.data.Dataset`

    :param folder_path: folder or list of folders containing sgf games
    :param history_n: specify input feature channels as (history_n + 1)*2

    .. note:: assuming same board size for all records
    """
    def __init__(self, folder_path: str | list[str], games: list = [],
                 history_n: int = 0):
        self.history_n = history_n
        total = 0
        if games:
            self.board_size = games[0].board_size
        if folder_path:
            if isinstance(folder_path, str):
                folder_path = [folder_path]
            self.board_size = None
            for folder in folder_path:
                loaded = load_sgf_games_in_folder(folder)
                if not loaded:
                    continue
                games += loaded
                if not self.board_size:
                    self.board_size = games[0].board_size
                elif self.board_size != loaded[0].board_size:
                    raise RuntimeError(f'board size mismatch {self.board_size}'
                                       f' != {loaded[0].board_size}')
        if games:
            self.winner = np.zeros(len(games), dtype=np.int8)
            for i, game in enumerate(games):
                total += len(game.moves)
                self.winner[i] = game.winner
                assert game.board_size == self.board_size
        self.total_moves = total
        if total > 0:
            #: concatenated moves
            self.game_moves = np.zeros(total, dtype=np.int16)
            #: game_id -> total moves before the game
            self.game_index = np.zeros(len(games) + 1, dtype=int)
            idx = 0
            for i, game in enumerate(games):
                self.game_index[i] = idx
                self.game_moves[idx:idx+len(game.moves)] = game.moves[:]
                idx += len(game.moves)
            self.game_index[len(games)] = idx
            assert total == idx

    def n_games(self) -> int:
        """number of games"""
        return len(self.winner)

    def input_channels(self) -> int:
        """number of input channels in network to train"""
        return (self.history_n + 1) * 2 + 1

    def to_game_move_pair(self, flat_idx) -> tuple[int, int]:
        """return pair of game id and move id"""
        gid = np.searchsorted(self.game_index, flat_idx, 'right') - 1
        move_id = flat_idx - self.game_index[gid]
        return gid, move_id

    def moves_view(self, game_id) -> np.ndarray:
        """return all moves in a game"""
        l, r = self.game_index[game_id], self.game_index[game_id + 1]
        return self.game_moves[l: r]

    def __len__(self):
        """return number of all moves in all games"""
        return self.total_moves

    def winner_sgn(self, game_id, move_id):
        """return 1 (-1) if result is win (loss)
        with respect for player to move, or 0 for draw
        """
        sgn = 1 if move_id % 2 == 0 else -1
        return self.winner[game_id] * sgn

    def __getitem__(self, flat_idx):
        """return a tuple of board_feature, move label, and winner label"""
        gid, move_id = self.to_game_move_pair(flat_idx)
        moves = self.moves_view(gid)
        state = cygo.State(self.board_size, max_history_n=self.history_n)
        cygo.apply_moves(state, moves[:move_id])
        xh = migo.features.history_n(state, self.history_n)
        xc = migo.features.color(state)
        x = np.vstack((xh, xc))
        y_move = moves[move_id]
        y_winner = self.winner_sgn(gid, move_id)
        return (torch.from_numpy(x),
                torch.Tensor([y_move]).long(),
                torch.Tensor([y_winner]))

    def to_img(self, flat_idx):
        """visualize specified data

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
        """visualize specified feature data

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
        """save all data to npz for future use without parsing sgf again"""
        np.savez_compressed(
            path,
            board_size=self.board_size,
            history_n=int(self.history_n),
            winner=self.winner,
            game_moves=self.game_moves,
            game_index=self.game_index,
            total_moves=self.total_moves,
        )

    @staticmethod
    def load_from(path):
        """load npz file (saved by `save_to`) to construct `SgfDataset`

        usage: `dataset = migo.dataset.SgfDataset('path-to-npz')`
        """
        objs = np.load(path)
        ret = SgfDataset('')
        ret.board_size = objs['board_size']
        ret.history_n = objs['history_n']
        ret.winner = objs['winner']
        ret.game_moves = objs['game_moves']
        ret.game_index = objs['game_index']
        ret.total_moves = objs['total_moves']
        return ret
