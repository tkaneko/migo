import copy
import json
import logging
import multiprocessing
import os
import os.path
import sys
import typing

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import tqdm.contrib.logging

import migo
import migo.cygo as cygo
import migo.features
import migo.network

global_config = {}
enable_terminal_score: bool = False

@click.group()
@click.option(
    '--log-level',
    type=click.Choice(
        ['debug', 'verbose', 'warning', 'quiet'], case_sensitive=False
    ),
)
def main(log_level):
    """run selfplay in parallel"""
    torch.set_float32_matmul_precision('high')

    FORMAT = '%(asctime)s %(levelname)s %(lineno)d %(message)s'
    level = logging.WARNING
    match log_level:
        case 'debug':
            level = logging.DEBUG
        case 'verbose':
            level = logging.INFO
        case 'warning':
            level = logging.WARNING
        case 'quiet':
            level = logging.CRITICAL
    logging.basicConfig(format=FORMAT, level=level)
    global_config['log_level'] = level


class GameStat:
    """manage statistics of a series of games"""

    def __init__(self):
        self.total_moves = 0
        self.win_count = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.pass_count = 0
        self.territory_count = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.takeall_count = {cygo.BLACK: 0, cygo.WHITE: 0}
        self.edge_occupancy = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.edge_occupancy_win = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.center_occupancy = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.center_occupancy_win = {
            cygo.BLACK: 0,
            cygo.WHITE: 0,
            cygo.EMPTY: 0,
        }
        self.merged = False
        self.primary_win = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}

    def add_games(self, new_games) -> None:
        """add new_games to self"""
        for game in new_games:
            self.add(game)

    def add(self, game) -> None:
        """add a game"""
        self.win_count[game.winner] += 1
        self.total_moves += len(game.moves)
        self.pass_count += (np.array(game.moves) == -1).sum()
        tb = np.maximum(game.territory, 0).sum()
        tw = -np.minimum(game.territory, 0).sum()
        self.territory_count[cygo.BLACK] += tb
        self.territory_count[cygo.WHITE] += tw
        if min(tb, tw) == 0:
            self.takeall_count[game.winner] += 1
        for zone_type in ['edge', 'center']:
            zone = migo.network.zone_plane(game.board_size, zone_type).flatten()
            territory = game.territory.flatten().astype(float)
            zb = np.dot(np.maximum(territory, 0), zone)
            zw = -np.dot(np.minimum(territory, 0), zone)
            ztotal = max(1, zone.sum())
            occupancy = (
                self.edge_occupancy
                if zone_type == 'edge'
                else self.center_occupancy
            )
            occupancy[cygo.BLACK] += zb / ztotal * 100
            occupancy[cygo.WHITE] += zw / ztotal * 100
            zwin = zb if game.winner == cygo.BLACK else zw
            occupancy_win = (
                self.edge_occupancy_win
                if zone_type == 'edge'
                else self.center_occupancy_win
            )
            occupancy_win[game.winner] += zwin / ztotal * 100

    def pass_ratio(self) -> float:
        """ratio of pass over all moves"""
        eps = 1e-8
        return (self.pass_count + eps) / (self.total_moves + eps)

    @property
    def black_wins(self) -> int:
        """number of games black won"""
        return self.win_count.get(cygo.BLACK, 0)

    @property
    def white_wins(self):
        """number of games white won"""
        return self.win_count.get(cygo.WHITE, 0)

    @property
    def n_games(self) -> int:
        """total number of games added"""
        return sum(self.win_count.values())

    @property
    def draws(self) -> int:
        """number of games draw"""
        return self.win_count[cygo.EMPTY]

    @property
    def black_win_ratio(self) -> float:
        """win ratio for black"""
        return (self.black_wins + self.draws / 2) / self.n_games

    @property
    def black_elo(self) -> float:
        """relative elo w.r.t win ratio for black"""
        return p2elo(self.black_win_ratio)

    def report(self) -> None:
        """show summary in stdout"""
        import tabulate

        bwin, wwin = self.black_wins, self.white_wins
        names = ['player', 'opponent'] if self.merged else ['black', 'white']
        lines = [
            ['wins', f'{bwin}', f'{wwin}'],
            [
                'takeall',
                f'{self.takeall_count[cygo.BLACK]}',
                f'{self.takeall_count[cygo.WHITE]}',
            ],
            ['elo', f'{self.black_elo:.1f}'],
            ['average moves', f'{self.total_moves / self.n_games:.1f}'],
            [
                'average territory',
                f'{self.territory_count[cygo.BLACK] / self.n_games:.1f}',
                f'{self.territory_count[cygo.WHITE] / self.n_games:.1f}',
            ],
        ]
        if sum(self.edge_occupancy.values()) != 0:
            lines.append(
                [
                    'edge occupancy (%)',
                    f'{self.edge_occupancy[cygo.BLACK] / self.n_games:.1f}',
                    f'{self.edge_occupancy[cygo.WHITE] / self.n_games:.1f}',
                ],
            )
            lines.append(
                [
                    'edge occupancy win (%)',
                    f'{self.edge_occupancy_win[cygo.BLACK] / max(bwin, 1):.1f}',
                    f'{self.edge_occupancy_win[cygo.WHITE] / max(wwin, 1):.1f}',
                ],
            )
        if sum(self.center_occupancy.values()) != 0:
            lines.append(
                [
                    'center occupancy (%)',
                    f'{self.center_occupancy[cygo.BLACK] / self.n_games:.1f}',
                    f'{self.center_occupancy[cygo.WHITE] / self.n_games:.1f}',
                ],
            )
            lines.append(
                [
                    'center occupancy win (%)',
                    f'{self.center_occupancy_win[cygo.BLACK] / max(bwin, 1):.1f}',
                    f'{self.center_occupancy_win[cygo.WHITE] / max(wwin, 1):.1f}',
                ],
            )
        print(tabulate.tabulate(lines, ['match'] + names, tablefmt='rst'))

    def merge(self, other: 'GameStat') -> None:
        """merge statistics"""
        b, w, e = cygo.BLACK, cygo.WHITE, cygo.EMPTY
        self.merged = True
        self.primary_win = {
            cygo.BLACK: self.win_count[b],
            cygo.WHITE: other.win_count[w],
        }
        self.win_count[b] += other.win_count[w]
        self.win_count[w] += other.win_count[b]
        self.win_count[e] += other.win_count[e]
        # follow win_count
        self.territory_count[b] += other.territory_count[w]
        self.territory_count[w] += other.territory_count[b]
        self.takeall_count[b] += other.takeall_count[w]
        self.takeall_count[w] += other.takeall_count[b]
        self.total_moves += other.total_moves
        self.edge_occupancy_win[b] += other.edge_occupancy_win[w]
        self.edge_occupancy_win[w] += other.edge_occupancy_win[b]
        self.edge_occupancy[b] += other.edge_occupancy[w]
        self.edge_occupancy[w] += other.edge_occupancy[b]
        self.center_occupancy_win[b] += other.center_occupancy_win[w]
        self.center_occupancy_win[w] += other.center_occupancy_win[b]
        self.center_occupancy[b] += other.center_occupancy[w]
        self.center_occupancy[w] += other.center_occupancy[b]


class TorchTRTInfer:
    """inference module to handle torch_tensorrt's script"""

    def __init__(self, path: str, device: str):
        logging.getLogger().setLevel(logging.CRITICAL)
        import torch_tensorrt  # ty: ignore[unresolved-import]

        with torch_tensorrt.logging.errors():  # ty: ignore[unresolved-attribute]
            if path.endswith('.ts'):
                self.trt_module = torch.jit.load(path)
            else:
                self.trt_module = torch.export.load(
                    path
                ).module()  # https://docs.pytorch.org/TensorRT/
        logging.getLogger().setLevel(global_config['log_level'])
        self.board_size = 9
        self.in_channels = 17

        self.device = device
        logging.info(f'loaded {path}')
        logging.debug(self.trt_module)
        cfg_path = os.path.splitext(path)[0] + '.json'
        self.with_aux_input = False
        if not os.path.exists(cfg_path):
            logging.warning(f'config {cfg_path} not found')
        else:
            with open(cfg_path) as f:
                cfg = json.loads(f.read())
            self.in_channels = cfg['in_channels']
            self.board_size = cfg['board_size']
            if 'with_aux_input' in cfg:
                self.with_aux_input = True
        logging.info(f'configured {self.board_size=} {self.in_channels=}')

    def infer(
        self, inputs: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """inference for a batch of inputs
        returns (
        (batch of main logins, [batch of aux logits]),
        (batch of main values),
        [batch of aux outputs]
        )
        """
        with torch.cuda.device(torch.device(self.device)):
            tensor = inputs.to(self.device).half()
            logits, values, *aux = self.trt_module(tensor)
        center = self.board_size**2 + 1
        logits_pair = [logits[:, :center], logits[:, center:]]
        return logits_pair, values, *aux


def state_features_py(state_list, history_n: int = 7, zone=None):
    """alternative (naive) implementation of state_features"""
    feature_dim = (history_n + 1) * 2 + 1
    board_size = state_list[0].board_size
    ret = np.empty((len(state_list), feature_dim, board_size, board_size))
    legals_relaxed = np.ones((len(state_list), board_size * board_size + 1))
    for i, state in enumerate(state_list):
        feature_history = migo.features.history_n(state, history_n)
        feature_turn = migo.features.color_black(state)
        if zone:
            item = np.vstack((feature_history, feature_turn, zone))
        else:
            item = np.vstack((feature_history, feature_turn))
        ret[i, :, :, :] = item
        stones = migo.features.board(state, dtype=np.int8)
        empties = 1 - (stones[0] + stones[1]).flatten()
        legals_relaxed[i, : board_size**2] = empties
    return ret, legals_relaxed


def state_features(
    state_list: list[cygo.State],
    *,
    in_channels: int,
    zone_list: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute features from a sequence of states using batch processing.

    :param state_list: A list of ``cygo.State`` objects representing the state history to be processed.
    :param in_channels: The number of input channels, used to determine the lookback period (history length).
    :param zone_list: An optional list of ``np.ndarray`` defining spatial zones for feature calculation. Defaults to ``None``.
    :return: A tuple containing the extracted features and associated auxiliary data or labels.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    history_n = (in_channels - 1) // 2 - 1
    # a, b = state_features_py(state_list, history_n, zone)
    if zone_list is not None:
        c, d = cygo.features.batch_features_with_zone(
            state_list, history_n, zone_list
        )
    else:
        c, d = cygo.features.batch_features(state_list, history_n)
    return c, d


def transformQ(
    nnQ: float | torch.Tensor,
    cvisit: int = 50,
    maxnb: int = 1,
    cscale: float = 1.0,
) -> float | torch.Tensor:
    """Converts values into equivalent logits, following Gumbel AlphaZero."""
    Q = nnQ / 2.0 + 0.5
    return (cvisit + maxnb) * cscale * Q


penalty_scale = 1000


class ModelQueue:
    """
    Interface for batch inference.

    :param model: The TorchTRTInfer instance used for performing inference.
    :type model: TorchTRTInfer
    :param batch_size: The threshold number of items in the queue to trigger inference.
    :type batch_size: int
    """

    def __init__(self, model: TorchTRTInfer, batch_size: int):
        self.model = model
        self.default_batch_size = batch_size
        self.input_state: list[cygo.State] = []
        self.input_zone = []
        # index in current batch, i.e., self.output[0][logits-values][out_idx]
        self.out_idx = 0
        self.output = []  # list of output batches
        self.add_gumbel_noise = False

    def set_gumbel_noise(self, enable: bool = True) -> None:
        """
        (Re)set whether to add Gumbel noise in logits.

        :param enable: Whether to enable the addition of Gumbel noise. Defaults to True.
        :type enable: bool
        :raises AssertionError: If there are items currently waiting in the input state queue.
        """
        assert not self.input_state
        self.add_gumbel_noise = enable

    def push(
        self, state: cygo.State, zone: typing.Optional[np.ndarray]
    ) -> None:
        """
        Push a new state and an optional zone into the processing queue.
        If the number of states reaches the default batch size, inference is executed.

        :param state: The state object to add to the queue.
        :type state: cygo.State
        :param zone: An optional numpy array representing the zone, defaults to None.
        :type zone: typing.Optional[np.ndarray]
        """
        self.input_state.append(state)
        if zone is not None:
            self.input_zone.append(zone)
        if len(self.input_state) >= self.default_batch_size:
            self.do_inference()

    def _adjust_values(self, values, others) -> tuple:
        """
        transform values suitable for gumbel player, i.e., w.r.t. its parent
        return (batch of values, batch of aux values or empty)
        """
        values = transformQ(-values)  # negative sign is for alternating player
        aux_v = torch.empty_like(values)  # ty: ignore[invalid-argument-type]
        if self.model.with_aux_input:
            _aux_p, aux_v = others
            aux_v = transformQ(-aux_v)
        return values, aux_v

    def _gumbel_noise(self, logits) -> torch.Tensor:
        """sample a vector for noise with the shape of given logits"""
        dist = torch.distributions.Gumbel(
            torch.zeros_like(logits), torch.ones_like(logits)
        )
        return dist.sample()

    def _run_model_with_noise(self, input_features, legals_relaxed) -> tuple:
        """returns (
        (batch of main logins, batch of aux logits),
        (batch of main values, batch of aux values)
        )
        """
        inputs = torch.from_numpy(input_features)
        legals_relaxed = torch.from_numpy(legals_relaxed).to(self.model.device)
        penalty = (1 - legals_relaxed.float()) * penalty_scale
        logits, values, *others = self.model.infer(inputs)
        # penalize illegal_moves
        logits[0] -= penalty
        noise = self._gumbel_noise(logits[0])
        logits[0] += noise
        if self.model.with_aux_input:
            logits[1] -= penalty
            logits[1] += noise
        values = self._adjust_values(values, others)
        return (logits, values)

    def _run_model(self, input_features) -> tuple:
        """returns (
        (batch of main logins, batch of aux logits),
        (batch of main values, batch of aux values)
        )
        """
        inputs = torch.from_numpy(input_features)
        logits, values, *others = self.model.infer(inputs)
        values = self._adjust_values(values, others)
        return (logits, values)

    def do_inference(self) -> None:
        """infer data in self.input_{state,zone} to append results in self.output"""
        if not self.input_state:
            raise RuntimeError('no data for inferernce')
        # logging.warning(f'do_inference {len(self.input_state)=}')
        # (batch_size, feature_dim, board_size, board_size)
        input_features, legals_relaxed = state_features(
            self.input_state,
            in_channels=self.model.in_channels,
            zone_list=self.input_zone or None,
        )

        if self.add_gumbel_noise:
            logits, values = self._run_model_with_noise(
                input_features, legals_relaxed
            )
        else:
            logits, values = self._run_model(input_features)
        ret = (
            [tensor.cpu() for tensor in logits],  # len: batch_size
            [tensor.cpu() for tensor in values],
        )
        self.output.append(ret)
        self.input_state = []
        self.input_zone = []

    def pop(self) -> list:
        """return tuple of torch.Tensors"""
        if not self.output:
            self.do_inference()
        # logging.warning(f'{len(self.output)=}')
        logits = self.output[0][0]
        values = self.output[0][1]
        assert self.out_idx < len(logits[0])
        # logging.warning(f'{len(logits)=}')
        # logging.warning(f'{len(values)=}')
        # logging.warning(f'{logits[0].shape=}')
        # logging.warning(f'{logits[1].shape=}')
        ret = [
            [logits[0][self.out_idx], logits[1][self.out_idx]],
            [values[0][self.out_idx], values[1][self.out_idx]],
        ]
        self.out_idx += 1
        if self.out_idx >= len(logits[0]):
            self.out_idx = 0
            self.output.pop(0)
        # logging.warning(f'{ret=}')
        return ret


class PlayerModel(typing.NamedTuple):
    queue: ModelQueue
    root_width: int
    aux_weight: float = 0
    primary_zone: typing.Optional[np.ndarray] = None
    opening_zone: typing.Optional[np.ndarray] = None
    opening_zone_limit: int = 0
    greedy: bool = False


class Game:
    """hold a state of an ongoing game"""

    def __init__(self, config: migo.SimpleRecord):
        self.record_template = config
        self.reset()

    def reset(self) -> None:
        """prepare new game"""
        self.game = copy.copy(self.record_template)
        self.game.moves = []
        self.next_move = None
        self.state = cygo.State(self.game.board_size, self.game.komi)
        self.move_dim = self.state.board_size**2 + 1
        self.illegal_count = 0
        self.last_zone = None

    def _push(self, player: PlayerModel, state: cygo.State) -> None:
        """push state and zone (may be None) into player's queue"""
        in_opening = len(self.game.moves) < player.opening_zone_limit
        # zone can be None
        zone = player.opening_zone if in_opening else player.primary_zone
        self.last_zone = zone
        player.queue.push(state, zone)

    def make_policy_request(self, player: PlayerModel) -> None:
        """organize inference at root"""
        self._push(player, self.state)

    def to_move(self, move_id) -> cygo.Move | cygo.Pass:
        """convert id in policy output to cygo.Move"""
        if move_id + 1 == self.move_dim:
            return cygo.Pass
        return cygo.Move.from_raw_value(move_id, self.state.board_size)

    def _sample(self, moves) -> int:
        """choose an action from self.moves

        Prerequisites: self.moves must be ordered by priority in advance.
        """
        for i, _move in enumerate(moves):
            sampled = moves[i]
            if sampled + 1 == self.move_dim or self.state.is_legal(
                self.to_move(sampled)
            ):
                return sampled
            # logging.info('illegalmove')
        self.illegal_count += 1
        return self.move_dim - 1  # as pass

    def recv_logits(self, player: PlayerModel) -> None:
        """receive inference results"""
        width = max(4, player.root_width)
        logits, _v = player.queue.pop()
        # aggregate logits for zone
        if self.last_zone is not None and self.last_zone.sum() > 0:
            logits[0] *= 1 - player.aux_weight
            logits[0] += logits[1] * player.aux_weight
        logits = logits[0]
        topk = torch.topk(logits.cpu(), width)
        self.moves = topk.indices.numpy()
        self.scores = topk.values.numpy()
        self.terminal_scores: list[None | float] = [None for _ in range(len(self.scores))]

    def make_value_request(self, player: PlayerModel) -> None:
        """organize inference for selected children"""
        for i in range(player.root_width):
            child = self.state.copy()
            move = self.moves[i]
            is_pass = (move + 1) == self.move_dim
            if is_pass or not self.state.is_legal(self.to_move(move)):
                pass2 = enable_terminal_score and is_pass and self.state.last_move.is_pass

                child.make_move(cygo.Pass)
                if not is_pass:  # illegal move
                    self.scores[i] = -penalty_scale
                elif pass2:
                    terminal_score = child.tromp_taylor_score(self.state.current_player)
                    terminal_score = transformQ(np.sign(terminal_score))
                    self.terminal_scores[i] = terminal_score # ty: ignore[invalid-assignment]
            else:
                child.make_move(move)
            self._push(player, child)

    def _swap_score_move(self, a: int, b: int) -> None:
        self.scores[[a, b]] = self.scores[[b, a]]
        self.moves[[a, b]] = self.moves[[b, a]]

    def recv_values(self, player: PlayerModel) -> None:
        """receive inference results"""
        for i in range(player.root_width):
            _, values = player.queue.pop()
            value, aux_value = values
            assert len(value) == 1 and len(aux_value) == 1
            if self.terminal_scores[i] is not None:
                value[0] += self.terminal_scores[i]
            if self.last_zone is not None and self.last_zone.sum() > 0:
                value[0] *= 1 - player.aux_weight
                value[0] += aux_value[0] * player.aux_weight
            self.scores[i] += value[0] # transformed in advance
            # maintain top3
            if i > 0 and self.scores[i] > self.scores[0]:
                self._swap_score_move(0, i)
            if i > 1 and self.scores[i] > self.scores[1]:
                self._swap_score_move(1, i)
            if i > 2 and self.scores[i] > self.scores[2]:
                self._swap_score_move(2, i)

    def advance_game(self) -> migo.SimpleRecord | None:
        """Executes an action based on the thought so far.

        Returns a game record if the game has completed; otherwise,
        returns None.

        Prerequisite: self.moves must be prepared in advance via the
        following sequence of calls: make_policy_request, recv_logits,
        make_value_request, and recv_values.
        """
        move = self.to_move(self._sample(self.moves))
        self.state.make_move(move)
        self.game.moves.append(move.raw() if move else -1)
        self.next_move = None

        # prepare next game if completed
        if self.game.moves[-2:] == [-1, -1]:  # ends with pass, pass
            # includes komi
            self.game.score = self.state.tromp_taylor_score(cygo.Color.BLACK)
            if self.game.score != 0:
                self.game.winner = (
                    cygo.BLACK if self.game.score > 0 else cygo.WHITE
                )
            else:
                self.game.winner = cygo.EMPTY
            self.game.territory = self.state.tromp_taylor_fill()
            ret = self.game
            self.last_illegals = self.illegal_count
            self.reset()
            return ret
        return None


def p2elo(p, eps=2e-4):
    """convert 1:1 win ratio into elo (difference)"""
    return -400 * np.log10((1 + eps) / (abs(p) + eps / 2) - 1)


class GameManager:
    """manage a sequence of games being played in parallel"""

    def __init__(
        self,
        template: migo.SimpleRecord,
        player: PlayerModel,
        *,
        parallel: int,
        complete_queue,
    ):
        """
        :param history_n: history length, as a part of input feature
        :param gumbel_root_width: player type, 0 for policy
        """
        self.board_size = template.board_size
        self.parallel = parallel
        self.complete_queue = complete_queue

        self.players = [player, player]  # same player by default
        self.template = template
        self.on_going = [Game(self.template) for _ in range(parallel)]
        self.completed = []
        self.total_steps = 0
        self.restart_waiting = set()
        self.total_completed = 0
        self.total_illegals = 0

    def set_white_player(self, *, player: PlayerModel) -> None:
        """configure white player different from black"""
        self.players[1] = player

    def step(self) -> None:
        """make each game one step forward"""
        player = self.players[self.total_steps % 2]
        if self.total_steps % 2 == 0:
            # align black player
            for id in self.restart_waiting:
                self.on_going[id].reset()
            self.restart_waiting.clear()
        # probe policy
        player.queue.set_gumbel_noise(
            enable=not player.greedy
        )  # True except when greedy
        for _i, game in enumerate(self.on_going):
            game.make_policy_request(player)
        for _i, game in enumerate(self.on_going):
            # four: keep backup to assure legal move
            game.recv_logits(player)
        # evaluate child node
        if player.root_width > 0:
            player.queue.set_gumbel_noise(enable=False)
            for _i, game in enumerate(self.on_going):
                game.make_value_request(player)
            for _i, game in enumerate(self.on_going):
                game.recv_values(player)
        # play
        for id, game in enumerate(self.on_going):
            completed = game.advance_game()
            if completed:
                self.tell_completed(completed)
                self.total_illegals += game.last_illegals
                if self.total_steps % 2 == 0:  # the next player is white
                    self.restart_waiting.add(id)
        self.total_steps += 1

    def tell_completed(
        self, game_or_none: migo.SimpleRecord | None = None
    ) -> None:
        """add a completed game if game_or_none, or flush queue otherwise"""
        if game_or_none:
            self.total_completed += 1
            self.completed.append(game_or_none)
        report_block_size = 1
        if game_or_none is None or len(self.completed) >= report_block_size:
            self.complete_queue.put(self.completed)
            self.completed = []

    def play_games(self, ngames: int) -> None:
        """step until completing ngames"""
        while self.total_completed < ngames:
            self.step()
        self.tell_completed()


def store_completed(
    completed: list[migo.SimpleRecord],
    output,
    *,
    history_n: int,
    ignore_opening_moves: int = 0,
    sgf_output='',
    zone_black='',
    zone_white='',
):
    """store completed games to dataset file"""
    # write npz
    dataset = migo.SgfDataset(
        games=completed,
        history_n=history_n,  # todo
        ignore_opening_moves=ignore_opening_moves,
    )
    if zone_black or zone_white:
        logging.info(f'storing {zone_black=} {zone_white=}')
        board_size = dataset.board_size
        zones = np.array(
            [
                migo.network.zone_plane(board_size, zone_black or 'null'),
                migo.network.zone_plane(board_size, zone_white or 'null'),
            ]
        )
        dataset = migo.ExtendedDataset.build_from(
            dataset,
            zones,
        )

    dataset.save_to(output)
    # write sgfs
    if sgf_output:
        for i, game in enumerate(completed):
            opath = f'{sgf_output}/{i}.sgf'
            with open(opath, 'w') as f:
                print(migo.record_to_sgf(game), file=f)
    if len(completed) == 1:
        game = completed[0]
        print(game.moves)
        state = cygo.State(game.board_size, game.komi)
        for move in game.moves:
            state.make_move(move)
        print(state)
        print(state.tromp_taylor_score(cygo.Color.BLACK))


def make_player(
    model: TorchTRTInfer,
    parallel: int,
    gumbel_root_width,
    aux_weight,
    enable_zone_after: int,
    zone_type: str | None,
    greedy: bool,
) -> PlayerModel:
    """
    return PlayerModel configured by specified arguments
    :param model: NN
    :param parallel: number of games played in parallel
    :param gumbel_root_width: width to try one-ply lookahead (0 for policy only)
    :param enable_zone_after: opening period to play without zone specification
    :param greedy: disable gumbel noise
    """
    queue = ModelQueue(model, parallel)
    primary_zone, opening_zone = None, None
    if model.with_aux_input:
        zone_type = zone_type or 'null'
        zone = migo.network.zone_plane(model.board_size, zone_type)
        primary_zone = zone
        if enable_zone_after > 0:
            opening_zone = migo.network.zone_plane(model.board_size, 'null')
        else:
            opening_zone = zone
    player = PlayerModel(
        queue,
        gumbel_root_width,
        aux_weight=aux_weight,
        opening_zone_limit=enable_zone_after,
        primary_zone=primary_zone,
        opening_zone=opening_zone,
        greedy=greedy,
    )
    return player


def task_play(
    queue: multiprocessing.Queue,
    model,
    width: int,
    greedy: bool,
    device: str,
    zone_type: str | None,
    aux_weight,
    games: int,
    parallel: int,
    komi: float,
    enable_zone_after,
) -> None:
    """play specified number of games"""
    os.close(sys.stdout.fileno())
    model = TorchTRTInfer(model, device)
    board_size = model.board_size
    config = migo.SimpleRecord(board_size=board_size, komi=komi)
    player = make_player(
        model, parallel, width, aux_weight, enable_zone_after, zone_type, greedy
    )
    logging.debug(f'{player=}')
    mgr = GameManager(
        config,
        player,
        parallel=parallel,
        complete_queue=queue,
    )
    mgr.play_games(games)
    # print(f'illegals {mgr.total_illegals}')


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', default='cuda:0', help='empty string for auto')
@click.option(
    '--width', type=int, default=8, help='root width for gumbel player'
)
@click.option(
    '--zone', type=click.Choice(migo.network.zone_names, case_sensitive=False)
)
@click.option(
    '--aux-weight',
    type=click.FloatRange(0, 1),
    default=0,
    help='weight for auxiliary values if zone',
)
@click.option('--greedy', is_flag=True, help='disable gumbel noise')
@click.option('--games', type=int, help='#games to play', default=128 * 8)
@click.option(
    '--parallel',
    type=int,
    help='#games to play simultaneously (gpu batchsize)',
    default=128,
)
@click.option(
    '--history-n',
    type=int,
    default=7,
    help='number of history stored in dataset',
)
@click.option(
    '--output', type=str, default='newgame.db', help='filename for output'
)
@click.option(
    '--sgf-output',
    type=click.Path(exists=True, file_okay=False),
    help='folder to store sgfs, ignored if "."',
    default='.',
)
@click.option(
    '--n-procs', type=int, help='#number of process spawned', default=8
)
@click.option(
    '--enable-zone-after',
    type=int,
    default=0,
    help='ignore zone at opening and exclude from labels in db',
)
@click.option('--komi', type=float, default=7.0, help='komi for black')
@click.option('--tqdm-position', type=int, help='position for tqdm', default=0)
def play(
    model: str | os.PathLike,
    width: int,
    zone: str | None,
    aux_weight: float,
    greedy: bool,
    games: int,
    device: str,
    parallel: int,
    history_n: int,
    output: str | os.PathLike,
    sgf_output: str | os.PathLike,
    n_procs: int,
    enable_zone_after: int,
    komi: float,
    tqdm_position: int,
):
    """Run selfplay by MODEL.
    Expects filename.ts for MODEL and look for filename.json for config.
    """
    if games % n_procs != 0:
        raise ValueError('please configure games as multiples of n_procs')
    complete_queue = multiprocessing.Queue()
    procs = [
        multiprocessing.Process(
            target=task_play,
            args=[
                complete_queue,
                model,
                width,
                greedy,
                device,
                zone,
                aux_weight,
                games // n_procs,
                parallel,
                komi,
                enable_zone_after,
            ],
        )
        for _ in range(n_procs)
    ]
    for i in range(n_procs):
        procs[i].start()

    game_stat = GameStat()
    completed_games = []
    with tqdm.contrib.logging.logging_redirect_tqdm():
        with tqdm.tqdm(
            total=games, leave=False, smoothing=0.01, position=tqdm_position
        ) as pbar:
            pbar.set_description(f'game {model}')
            while len(completed_games) < games:
                new_games = complete_queue.get()
                completed_games += new_games
                pbar.update(len(new_games))
                game_stat.add_games(new_games)
    for i in range(n_procs):
        procs[i].join()
    if sgf_output == '.':
        sgf_output = ''
    store_completed(
        completed_games,
        output,
        history_n=history_n,
        ignore_opening_moves=enable_zone_after,
        sgf_output=sgf_output,
        zone_black=zone,
        zone_white=zone,
    )
    if games > 1 and tqdm_position == 0:
        game_stat.report()
    complete_queue.close()
    complete_queue.cancel_join_thread()


def task_match(
    queue: multiprocessing.Queue,
    model_a,
    width_a,
    device_a,
    zone_a,
    aux_weight_a,
    enable_zone_after_a,
    model_b,
    width_b,
    device_b,
    zone_b,
    aux_weight_b,
    enable_zone_after_b,
    games: int,
    parallel: int,
    komi: float,
):
    """play specified number of games by black player (a) and white player (b)"""
    os.close(sys.stdout.fileno())
    model_a = TorchTRTInfer(model_a, device_a)
    model_b = TorchTRTInfer(model_b, device_b)
    board_size = model_a.board_size
    config = migo.SimpleRecord(board_size=board_size, komi=komi)
    player_a = make_player(
        model_a,
        parallel,
        width_a,
        aux_weight_a,
        enable_zone_after_a,
        zone_a,
        greedy=False,
    )

    mgr = GameManager(
        config,
        player_a,
        parallel=parallel,
        complete_queue=queue,
    )
    player_b = make_player(
        model_b,
        parallel,
        width_b,
        aux_weight_b,
        enable_zone_after_b,
        zone_b,
        greedy=False,
    )
    mgr.set_white_player(player=player_b)
    mgr.play_games(games)


def do_match(
    model_a,
    width_a,
    device_a,
    zone_a,
    aux_weight_a,
    enable_zone_after_a,
    model_b,
    width_b,
    device_b,
    zone_b,
    aux_weight_b,
    enable_zone_after_b,
    games: int,
    parallel: int,
    n_procs: int,
    komi: float,
    *,
    tqdm_position=0,
    tqdm_clear=False,
) -> tuple[list, GameStat]:
    """play match of player (a) and white player (b)

    :param n_procs: number of game managers (i.e., cuda streams) run in parallel
    """
    complete_queue = multiprocessing.Queue()
    procs = [
        multiprocessing.Process(
            target=task_match,
            args=[
                complete_queue,
                model_a,
                width_a,
                device_a,
                zone_a,
                aux_weight_a,
                enable_zone_after_a,
                model_b,
                width_b,
                device_b,
                zone_b,
                aux_weight_b,
                enable_zone_after_b,
                games // n_procs,
                parallel,
                komi,
            ],
        )
        for _ in range(n_procs)
    ]

    for i in range(n_procs):
        procs[i].start()

    completed_games = []
    game_stat = GameStat()
    with tqdm.contrib.logging.logging_redirect_tqdm():
        with tqdm.tqdm(
            total=games,
            smoothing=0.01,
            position=tqdm_position,
            leave=not tqdm_clear,
        ) as pbar:
            while len(completed_games) < games:
                new_games = complete_queue.get()
                completed_games += new_games
                pbar.update(len(new_games))
                game_stat.add_games(new_games)
    for i in range(n_procs):
        procs[i].join()
    complete_queue.close()
    complete_queue.cancel_join_thread()
    return completed_games, game_stat


@main.command(context_settings={'show_default': True})
@click.argument('model-a', type=click.Path(exists=True, dir_okay=False))
@click.option('--device-a', default='cuda:0', help='empty string for auto')
@click.option('--width-a', type=int, default=8, help='root width for gumbel')
@click.option(
    '--zone-a', type=click.Choice(migo.network.zone_names, case_sensitive=False)
)
@click.option(
    '--aux-weight-a',
    type=click.FloatRange(0, 1),
    default=0,
    help='weight for auxiliary values in player a',
)
@click.option(
    '--enable-zone-after-a',
    type=int,
    default=0,
    help='ignore zone at opening and exclude from labels in db',
)
@click.argument('model-b', type=click.Path(exists=True, dir_okay=False))
@click.option('--device-b', default='cuda:0', help='empty string for auto')
@click.option('--width-b', type=int, default=8, help='root width for gumbel')
@click.option(
    '--zone-b', type=click.Choice(migo.network.zone_names, case_sensitive=False)
)
@click.option(
    '--aux-weight-b',
    type=click.FloatRange(0, 1),
    default=0,
    help='weight for auxiliary values in player b',
)
@click.option(
    '--enable-zone-after-b',
    type=int,
    default=0,
    help='ignore zone at opening and exclude from labels in db',
)
@click.option('--games', type=int, help='#games to play', default=10)
@click.option(
    '--parallel', type=int, help='#games to play simultaneously', default=128
)
@click.option(
    '--n-procs', type=int, help='#number of process spawned', default=8
)
@click.option('--output', type=str, default='', help='filename for output')
@click.option(
    '--sgf-output',
    type=click.Path(exists=True, file_okay=False),
    help='folder to store sgfs, ignored if "."',
    default='.',
)
@click.option('--history-n', type=int, default=7, help='history planes for db')
@click.option('--komi', type=float, default=7.0, help='komi for black')
@click.option('--tqdm-position', type=int, help='position for tqdm', default=0)
@click.option('--tqdm-clear', is_flag=True, help='erase pbar at end')
def match(
    model_a,
    width_a,
    device_a,
    zone_a,
    aux_weight_a,
    enable_zone_after_a,
    model_b,
    width_b,
    device_b,
    zone_b,
    aux_weight_b,
    enable_zone_after_b,
    games,
    parallel,
    n_procs,
    output,
    sgf_output,
    history_n,
    komi,
    tqdm_position,
    tqdm_clear,
):
    """Play match of MODEL_A (black) v.s. MODEL_B (white)."""

    completed_games, game_stat = do_match(
        model_a,
        width_a,
        device_a,
        zone_a,
        aux_weight_a,
        enable_zone_after_a,
        model_b,
        width_b,
        device_b,
        zone_b,
        aux_weight_b,
        enable_zone_after_b,
        games,
        parallel,
        n_procs,
        komi,
        tqdm_position=tqdm_position,
        tqdm_clear=tqdm_clear,
    )

    if sgf_output == '.':
        sgf_output = ''
    if output or sgf_output:
        store_completed(
            completed_games,
            output,
            history_n=history_n,
            ignore_opening_moves=max(enable_zone_after_a, enable_zone_after_b),
            sgf_output=sgf_output,
            zone_black=zone_a,
            zone_white=zone_b,
        )
    game_stat.report()


def do_evalelo(
    model_a,
    width_a,
    device_a,
    zone_a,
    aux_weight_a,
    enable_zone_after_a,
    models,
    width,
    device,
    zone,
    aux_weight,
    enable_zone_after,
    games,
    parallel,
    n_procs,
    komi,
):
    elos = []
    for model in models:
        _, game_stat_rev = do_match(
            model_a,
            width_a,
            device_a,
            zone_a,
            aux_weight_a,
            enable_zone_after_a,
            model,
            width,
            device,
            zone,
            aux_weight,
            enable_zone_after,
            games,
            parallel,
            n_procs,
            komi,
        )
        _, game_stat = do_match(
            model,
            width,
            device,
            zone,
            aux_weight,
            enable_zone_after,
            model_a,
            width_a,
            device_a,
            zone_a,
            aux_weight_a,
            enable_zone_after_a,
            games,
            parallel,
            n_procs,
            komi,
        )
        game_stat.merge(game_stat_rev)
        print(model)
        game_stat.report()
        elos.append(game_stat.black_elo)
    return elos


@main.command(context_settings={'show_default': True})
@click.argument(
    'model-a',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option('--device-a', default='cuda:0', help='empty string for auto')
@click.option('--width-a', type=int, default=8, help='root width for gumbel')
@click.option(
    '--zone-a', type=click.Choice(migo.network.zone_names, case_sensitive=False)
)
@click.option(
    '--aux-weight-a',
    type=click.FloatRange(0, 1),
    default=0,
    help='weight for auxiliary values in base player',
)
@click.option(
    '--enable-zone-after-a',
    type=int,
    default=0,
    help='ignore zone at opening and exclude from labels in db',
)
@click.argument(
    'models', type=click.Path(exists=True, dir_okay=False), nargs=-1
)
@click.option('--device', default='cuda:0', help='empty string for auto')
@click.option('--width', type=int, default=8, help='root width for gumbel')
@click.option(
    '--zone',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option(
    '--zone2',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option(
    '--zone3',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option(
    '--zone4',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option(
    '--aux-weight',
    type=click.FloatRange(0, 1),
    default=0,
    help='weight for auxiliary values in players',
)
@click.option(
    '--enable-zone-after',
    type=int,
    default=0,
    help='ignore zone at opening and exclude from labels in db',
)
@click.option('--games', type=int, help='#games to play', default=2000)
@click.option(
    '--parallel', type=int, help='#games to play simultaneously', default=128
)
@click.option(
    '--n-procs', type=int, help='#number of process spawned', default=8
)
@click.option('--komi', type=float, default=7.0, help='komi for black')
@click.option('--savefig', default='', help='filename for output')
def evalelo(
    model_a,
    width_a,
    device_a,
    zone_a,
    aux_weight_a,
    enable_zone_after_a,
    models,
    width,
    device,
    zone,
    zone2,
    zone3,
    zone4,
    aux_weight,
    enable_zone_after,
    games,
    parallel,
    n_procs,
    komi,
    savefig,
):
    """evaluate relative elo of MODELS playing against MODEL_A"""
    if (model_a.endswith('.pth')):
        logging.error(f'filename.ts is expected but received {model_a}')
        return 1
    results = []
    for i, zone_now in enumerate([zone, zone2, zone3, zone4]):
        if i > 0 and not zone_now:
            continue
        elos = do_evalelo(
            model_a,
            width_a,
            device_a,
            zone_a,
            aux_weight_a,
            enable_zone_after_a,
            models,
            width,
            device,
            zone_now,
            aux_weight,
            enable_zone_after,
            games,
            parallel,
            n_procs,
            komi,
        )
        results.append((zone_now, elos))

    with np.printoptions(formatter={'float': '{: 0.1f}'.format}):
        for zone_now, elos in results:
            print('Zone ', zone_now)
            print(np.array(elos))
    if savefig:
        plot_elo_series(models, results, savefig)


def plot_elo_series(models, results, output, ylim=(-600, 200), figsize=(7, 3)):
    # assumes mname-[0-9][0-9][0-9][0-9].ts
    ages = [int(_[-7:-3]) for _ in models]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for _i, (lbl, series) in enumerate(results):
        ax.plot(  # stepc
            ages, np.array(series), label=lbl or 'base'
        )
    ax.axhline(0, alpha=0.7, ls='-.')
    ax.legend()
    ax.set_xlabel('age')
    ax.set_ylabel('elo')
    ax.set_ylim(*ylim)
    fig.savefig(output)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    main()
