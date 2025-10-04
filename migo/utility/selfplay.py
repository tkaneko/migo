import migo
import migo.network
import migo.features
import cygo
import recordclass
import click
import torch
import tqdm
import tqdm.contrib.logging
import numpy as np
import logging
import copy
import multiprocessing
import os
import os.path
import sys
import json


global_config = {}


@click.group()
@click.option('--log-level',
              type=click.Choice(['debug', 'verbose', 'warning', 'quiet'],
                                case_sensitive=False))
def main(log_level):
    """run selfplay in parallel"""
    torch.set_float32_matmul_precision('high')

    FORMAT = '%(asctime)s %(levelname)s %(lineno)d %(message)s'
    level = logging.WARNING
    match log_level:
        case 'debug': level = logging.DEBUG
        case 'verbose': level = logging.INFO
        case 'warning': level = logging.WARNING
        case 'quiet': level = logging.CRITICAL
    logging.basicConfig(format=FORMAT, level=level)
    global_config['log_level'] = level


class GameStat:
    def __init__(self):
        self.total_moves = 0
        self.win_count = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.pass_count = 0
        self.territory_count = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.takeall_count = {cygo.BLACK: 0, cygo.WHITE: 0}
        self.edge_occupancy = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.edge_occupancy_win = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.center_occupancy = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.center_occupancy_win = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}
        self.merged = False
        self.primary_win = {cygo.BLACK: 0, cygo.WHITE: 0, cygo.EMPTY: 0}

    def add_games(self, new_games):
        for game in new_games:
            self.add(game)

    def add(self, game):
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
                if zone_type == 'edge' else self.center_occupancy
            )
            occupancy[cygo.BLACK] += zb / ztotal * 100
            occupancy[cygo.WHITE] += zw / ztotal * 100
            zwin = zb if game.winner == cygo.BLACK else zw
            occupancy_win = (
                self.edge_occupancy_win
                if zone_type == 'edge' else self.center_occupancy_win
            )
            occupancy_win[game.winner] += zwin / ztotal * 100

    def pass_ratio(self):
        eps = 1e-8
        return (self.pass_count + eps) / (self.total_moves + eps)

    @property
    def black_wins(self):
        return self.win_count.get(cygo.BLACK, 0)

    @property
    def white_wins(self):
        return self.win_count.get(cygo.WHITE, 0)

    @property
    def n_games(self):
        return sum(self.win_count.values())

    @property
    def draws(self):
        return self.win_count[cygo.EMPTY]

    @property
    def black_win_ratio(self):
        return (self.black_wins + self.draws / 2) / self.n_games

    @property
    def black_elo(self):
        return p2elo(self.black_win_ratio)

    def report(self):
        import tabulate
        bwin, wwin = self.black_wins, self.white_wins
        names = ['player', 'opponent'] if self.merged else ['black', 'white']
        lines = [
            ['wins', f'{bwin}', f'{wwin}'],
            ['takeall',
             f'{self.takeall_count[cygo.BLACK]}',
             f'{self.takeall_count[cygo.WHITE]}'],
            ['elo', f'{self.black_elo:.1f}'],
            ['average moves', f'{self.total_moves / self.n_games:.1f}'],
            ['average territory',
             f'{self.territory_count[cygo.BLACK] / self.n_games:.1f}',
             f'{self.territory_count[cygo.WHITE] / self.n_games:.1f}'],
        ]
        if sum(self.edge_occupancy.values()) != 0:
            lines.append([
                'edge occupancy (%)',
                f'{self.edge_occupancy[cygo.BLACK] / self.n_games:.1f}',
                f'{self.edge_occupancy[cygo.WHITE] / self.n_games:.1f}',
            ],)
            lines.append([
                'edge occupancy win (%)',
                f'{self.edge_occupancy_win[cygo.BLACK] / max(bwin, 1):.1f}',
                f'{self.edge_occupancy_win[cygo.WHITE] / max(wwin, 1):.1f}'
            ],)
        if sum(self.center_occupancy.values()) != 0:
            lines.append([
                'center occupancy (%)',
                f'{self.center_occupancy[cygo.BLACK] / self.n_games:.1f}',
                f'{self.center_occupancy[cygo.WHITE] / self.n_games:.1f}',
            ],)
            lines.append([
                'center occupancy win (%)',
                f'{self.center_occupancy_win[cygo.BLACK] / max(bwin, 1):.1f}',
                f'{self.center_occupancy_win[cygo.WHITE] / max(wwin, 1):.1f}'
            ],)
        print(tabulate.tabulate(lines, ['match']+names, tablefmt='rst'))

    def merge(self, other):
        b, w, e = cygo.BLACK, cygo.WHITE, cygo.EMPTY
        self.merged = True
        self.primary_win = {
            cygo.BLACK: self.win_count[b], cygo.WHITE: other.win_count[w]
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
    '''inference module to handle torch_tensorrt's script'''
    def __init__(self, path: str, device: str):
        logging.getLogger().setLevel(logging.CRITICAL)
        import torch_tensorrt
        with torch_tensorrt.logging.errors():
            self.trt_module = torch.jit.load(path)
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

    def infer(self, inputs: torch.Tensor):
        '''inference for a batch of inputs'''
        with torch.cuda.device(torch.device(self.device)):
            tensor = inputs.to(self.device).half()
            logits, values, *aux = self.trt_module(tensor)
        center = self.board_size**2+1
        logits = [logits[:, :center], logits[:, center:]]
        return logits, values, *aux


def state_features_py(state_list, history_n: int = 7, zone=None):
    feature_dim = (history_n + 1)*2 + 1
    board_size = state_list[0].board_size
    ret = np.empty(
        (len(state_list), feature_dim, board_size, board_size)
    )
    legals_relaxed = np.ones(
        (len(state_list), board_size * board_size + 1)
    )
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
        legals_relaxed[i, :board_size**2] = empties
    return ret, legals_relaxed


def state_features(state_list, *, in_channels: int, zone_list=None):
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
        nnQ: float, cvisit: int = 50, maxnb: int = 1, cscale: float = 1.0
) -> float:
    Q = nnQ/2.0 + 0.5
    return (cvisit + maxnb) * cscale * Q


penalty_scale = 1000


class ModelQueue:
    '''interface for batch inference'''
    def __init__(self, model, batch_size: int):
        self.model = model
        self.default_batch_size = batch_size
        self.input_state = []
        self.input_zone = []
        # index in current batch, i.e., self.output[0][logits-values][out_idx]
        self.out_idx = 0
        self.output = []        # list of output batches
        self.add_gumbel_noise = False

    def set_gumbel_noise(self):
        assert not self.input_state
        self.add_gumbel_noise = True

    def unset_gumbel_noise(self):
        assert not self.input_state
        self.add_gumbel_noise = False

    def push(self, state, zone):
        self.input_state.append(state)
        if zone is not None:
            self.input_zone.append(zone)
        if len(self.input_state) >= self.default_batch_size:
            self.do_inference()

    def _adjust_values(self, values, others):
        '''
        transform values suitable for gumbel player, i.e., w.r.t. its parent
        return (batch of values, batch of aux values or empty)
        '''
        values = transformQ(-values)
        aux_v = torch.empty_like(values)
        if self.model.with_aux_input:
            _aux_p, aux_v = others
            aux_v = transformQ(-aux_v)
        return values, aux_v

    def _gumbel_noise(self, logits):
        dist = torch.distributions.Gumbel(
            torch.zeros_like(logits), torch.ones_like(logits)
        )
        return dist.sample()

    def _run_model_with_noise(self, input_features, legals_relaxed):
        '''returns (
        (batch of main logins, batch of aux logits),
        (batch of main values, batch of aux values)
        )
        '''
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

    def _run_model(self, input_features):
        '''returns (
        (batch of main logins, batch of aux logits),
        (batch of main values, batch of aux values)
        )
        '''
        inputs = torch.from_numpy(input_features)
        logits, values, *others = self.model.infer(inputs)
        values = self._adjust_values(values, others)
        return (logits, values)

    def do_inference(self):
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
            [tensor.cpu() for tensor in logits],
            [tensor.cpu() for tensor in values],
        )
        self.output.append(ret)
        self.input_state = []
        self.input_zone = []

    def pop(self):
        '''return tuple of torch.Tensors (may resident in gpu memory)'''
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


class Game:
    def __init__(self, config):
        self.record_template = config
        self.reset()

    def reset(self):
        '''prepare new game'''
        self.game = copy.copy(self.record_template)
        self.game.moves = []
        self.next_move = None
        self.state = cygo.State(self.game.board_size, self.game.komi)
        self.move_dim = self.state.board_size ** 2 + 1
        self.illegal_count = 0
        self.last_zone = None

    def _push(self, player, state):
        in_opening = len(self.game.moves) < player.opening_zone_limit
        # zone can be None
        zone = player.opening_zone if in_opening else player.primary_zone
        self.latest_zone = zone
        player.queue.push(state, zone)

    def make_policy_request(self, player):
        '''organize inference at root'''
        self._push(player, self.state)

    def to_move(self, move_id) -> cygo.Move:
        '''convert id in policy output to cygo.Move'''
        if move_id + 1 == self.move_dim:
            return cygo.Pass
        return cygo.Move.from_raw_value(move_id, self.state.board_size)

    def sample(self, moves) -> int:
        '''choose an action'''
        for i, move in enumerate(moves):
            sampled = moves[i]
            if sampled + 1 == self.move_dim \
               or self.state.is_legal(self.to_move(sampled)):
                return sampled
            # logging.info('illegalmove')
        self.illegal_count += 1
        return self.move_dim - 1    # as pass

    def recv_logits(self, player, width) -> bool:
        '''receive inference results'''
        logits, _v = player.queue.pop()
        # aggregate logits for zone
        if self.last_zone is not None and self.last_zone.sum() > 0:
            logits[0] *= (1 - player.aux_weight)
            logits[0] += logits[1] * player.aux_weight
        logits = logits[0]
        topk = torch.topk(logits.cpu(), width)
        self.moves = topk.indices.numpy()
        self.scores = topk.values.numpy()

    def make_value_request(self, player, width):
        '''organize inference for selected children'''
        for i in range(width):
            child = self.state.copy()
            move = self.moves[i]
            is_pass = (move + 1) == self.move_dim
            if is_pass \
               or not self.state.is_legal(self.to_move(move)):
                child.make_move(cygo.Pass)
                if not is_pass:  # illegal move
                    self.scores[i] = -penalty_scale
            else:
                child.make_move(move)
            self._push(player, child)

    def _swap_score_move(self, a, b):
        self.scores[[a, b]] = self.scores[[b, a]]
        self.moves[[a, b]] = self.moves[[b, a]]

    def recv_values(self, player, width):
        '''receive inference results'''
        for i in range(width):
            _, values = player.queue.pop()
            value, aux_value = values
            if self.last_zone is not None and self.last_zone.sum() > 0:
                value *= (1 - player.aux_weight)
                value += aux_value * player.aux_weight
            self.scores[i] += value[0].item()  # transformed in advance
            # maintain top3
            if i > 0 and self.scores[i] > self.scores[0]:
                self._swap_score_move(0, i)
            if i > 1 and self.scores[i] > self.scores[1]:
                self._swap_score_move(1, i)
            if i > 2 and self.scores[i] > self.scores[2]:
                self._swap_score_move(2, i)

    def step(self) -> migo.SimpleRecord | None:
        '''play an action.

        return a game record if completed or None otherwise
        '''
        move = self.to_move(self.sample(self.moves))
        self.state.make_move(move)
        self.game.moves.append(move.raw() if move else -1)
        self.next_move = None
        if self.game.moves[-2:] == [-1, -1]:
            # includes komi
            self.game.score = self.state.tromp_taylor_score(cygo.Color.BLACK)
            if self.game.score != 0:
                self.game.winner = cygo.BLACK \
                    if self.game.score > 0 else cygo.WHITE
            else:
                self.game.winner = cygo.EMPTY
            self.game.territory = self.state.tromp_taylor_fill()
            ret = self.game
            self.last_illegals = self.illegal_count
            self.reset()
            return ret
        return None


PlayerModel = recordclass.recordclass(
    'PlayerModel', (
        'queue', 'root_width',
        # optional
        'aux_weight', 'primary_zone', 'opening_zone', 'opening_zone_limit',
     ))


def p2elo(p, eps=2e-4):
    return -400 * np.log10((1+eps)/(abs(p)+eps/2)-1)


class GameManager:
    '''manage a sequence of games being played in parallel'''
    def __init__(self, model,
                 template: migo.SimpleRecord,
                 player: PlayerModel,
                 n_games: int, *,
                 parallel: int,
                 complete_queue):
        '''
        :param history_n: history length, as a part of input feature
        :param gumbel_root_width: player type, 0 for policy
        '''
        self.board_size = template.board_size
        self.n_games = n_games
        self.parallel = parallel
        self.complete_queue = complete_queue

        self.players = [player, player]  # same player by default
        self.template = template
        self.on_going = [Game(self.template) for _ in range(parallel)]
        self.completed = []
        self.total_steps = 0
        self.should_restart = set()
        self.total_completed = 0
        self.total_illegals = 0

    def set_white_player(self, *,
                         player: PlayerModel):
        '''configure white player different from black'''
        self.players[1] = player

    def step(self):
        '''make each game one step forward'''
        player = self.players[self.total_steps % 2]
        if self.total_steps % 2 == 0:
            # align black player
            for id in self.should_restart:
                self.on_going[id].reset()
            self.should_restart.clear()
        # probe policy
        player.queue.set_gumbel_noise()
        for i, game in enumerate(self.on_going):
            game.make_policy_request(player)
        for i, game in enumerate(self.on_going):
            # four: keep backup to assure legal move
            game.recv_logits(player, max(4, player.root_width))
        # evaluate child node
        if player.root_width > 0:
            player.queue.unset_gumbel_noise()
            for i, game in enumerate(self.on_going):
                game.make_value_request(player, player.root_width)
            for i, game in enumerate(self.on_going):
                game.recv_values(player, player.root_width)
        # play
        for id, game in enumerate(self.on_going):
            completed = game.step()
            if completed:
                self.tell_completed(completed)
                self.total_illegals += game.last_illegals
                if self.total_steps % 2 == 0:  # the next player is white
                    self.should_restart.add(id)
        self.total_steps += 1

    def tell_completed(self, game_or_none=None):
        if game_or_none:
            self.total_completed += 1
            self.completed.append(game_or_none)
        report_block_size = 1
        if game_or_none is None or len(self.completed) >= report_block_size:
            self.complete_queue.put(self.completed)
            self.completed = []

    def play_games(self, ngames):
        '''step until completing ngames'''
        while self.total_completed < ngames:
            self.step()
        self.tell_completed()


def store_completed(completed, output, *,
                    history_n: int,
                    ignore_opening_moves: int = 0,
                    sgf_output='',
                    zone_black='', zone_white=''):
    # write npz
    dataset = migo.SgfDataset(
        games=completed,
        history_n=history_n,  # todo
        ignore_opening_moves=ignore_opening_moves
    )
    if zone_black or zone_white:
        logging.info(f'storing {zone_black=} {zone_white=}')
        board_size = dataset.board_size
        zones = np.array([
            migo.network.zone_plane(board_size, zone_black or 'null'),
            migo.network.zone_plane(board_size, zone_white or 'null'),
        ])
        dataset = migo.ExtendedDataset.build_from(
            dataset, zones,
            batch_with_collate=True,
        )

    dataset.save_to(output)
    # write sgfs
    if sgf_output:
        for i, game in enumerate(completed):
            opath = f'{sgf_output}/{i}.sgf'
            with open(opath, 'w') as f:
                print(migo.record_to_sgf(game), file=f)


def make_player(model, parallel, gumbel_root_width, aux_weight,
                enable_zone_after, zone_type):
    queue = ModelQueue(model, parallel)
    player = PlayerModel(
        queue, gumbel_root_width,
        aux_weight=aux_weight,
        opening_zone_limit=enable_zone_after,
    )
    if model.with_aux_input:
        zone_type = zone_type or 'null'
        zone = migo.network.zone_plane(
            model.board_size, zone_type
        )
        player.primary_zone = zone
        if enable_zone_after > 0:
            player.opening_zone = migo.network.zone_plane(
                model.board_size, 'null'
            )
        else:
            player.opening_zone = zone
    return player


def task_play(
        queue,
        model, width, device, zone_type, aux_weight,
        games, parallel,
        enable_zone_after,
):
    os.close(sys.stdout.fileno())
    model = TorchTRTInfer(model, device)
    board_size = 9
    config = migo.SimpleRecord(board_size=board_size, komi=7.0)
    player = make_player(model, parallel, width, aux_weight,
                         enable_zone_after, zone_type)
    logging.debug(f'{player=}')
    mgr = GameManager(
        model, config, player, games,
        parallel=parallel,
        complete_queue=queue,
    )
    mgr.play_games(games)
    # print(f'illegals {mgr.total_illegals}')


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', default='cuda:0', help='empty string for auto')
@click.option('--width', type=int, default=8,
              help='root width for gumbel player')
@click.option('--zone',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
@click.option('--aux-weight', type=click.FloatRange(0, 1), default=0,
              help='weight for auxiliary values if zone'
              )
@click.option('--games', type=int, help='#games to play',
              default=128*8)
@click.option('--parallel', type=int,
              help='#games to play simultaneously (gpu batchsize)',
              default=128)
@click.option(
    '--history-n', type=int, default=7,
    help='number of history stored in dataset'
)
@click.option('--output', type=str, default='newgame.db',
              help='filename for output')
@click.option('--sgf-output', type=click.Path(exists=True, file_okay=False),
              help='folder to store sgfs, ignored if "."',
              default='.')
@click.option('--n-procs', type=int, help='#number of process spawned',
              default=8)
@click.option(
    '--enable-zone-after', type=int, default=0,
    help='ignore zone at opening and exclude from labels in db'
)
@click.option('--tqdm-position', type=int, help='position for tqdm',
              default=0)
def play(model, width, zone, aux_weight,
         games, device, parallel, history_n,
         output, sgf_output, n_procs,
         enable_zone_after, tqdm_position):
    """Run selfplay by MODEL.
    Expects filename.ts for MODEL and look for filename.json for config.
    """
    if games % n_procs != 0:
        raise ValueError('please configure games as multiples of n_procs')
    complete_queue = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=task_play, args=[
        complete_queue,
        model, width, device, zone, aux_weight,
        games//n_procs, parallel, enable_zone_after,
    ]) for _ in range(n_procs)]
    for i in range(n_procs):
        procs[i].start()

    game_stat = GameStat()
    completed_games = []
    with tqdm.contrib.logging.logging_redirect_tqdm():
        with tqdm.tqdm(
                total=games, leave=False, smoothing=0.01,
                position=tqdm_position
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
        completed_games, output, history_n=history_n,
        ignore_opening_moves=enable_zone_after,
        sgf_output=sgf_output,
        zone_black=zone, zone_white=zone,
    )
    game_stat.report()
    complete_queue.close()
    # complete_queue.cancel_join_thread()


def task_match(
        queue,
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        model_b, width_b, device_b, zone_b, aux_weight_b, enable_zone_after_b,
        games, parallel,
):
    os.close(sys.stdout.fileno())
    model_a = TorchTRTInfer(model_a, device_a)
    model_b = TorchTRTInfer(model_b, device_b)
    board_size = 9
    config = migo.SimpleRecord(board_size=board_size, komi=7.0)
    player_a = make_player(
        model_a, parallel, width_a, aux_weight_a, enable_zone_after_a, zone_a
    )

    mgr = GameManager(
        model_a, config, player_a, games,
        parallel=parallel,
        complete_queue=queue,
    )
    player_b = make_player(
        model_b, parallel, width_b, aux_weight_b, enable_zone_after_b, zone_b
    )
    mgr.set_white_player(player=player_b)
    mgr.play_games(games)


def do_match(
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        model_b, width_b, device_b, zone_b, aux_weight_b, enable_zone_after_b,
        games, parallel, n_procs, *,
        tqdm_position=0, tqdm_clear=False, zone_for_score=''):
    complete_queue = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=task_match, args=[
        complete_queue,
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        model_b, width_b, device_b, zone_b, aux_weight_b, enable_zone_after_b,
        games//n_procs, parallel
    ]) for _ in range(n_procs)]

    for i in range(n_procs):
        procs[i].start()

    completed_games = []
    game_stat = GameStat()
    with tqdm.contrib.logging.logging_redirect_tqdm():
        with tqdm.tqdm(total=games, smoothing=0.01,
                       position=tqdm_position, leave=not tqdm_clear) as pbar:
            while len(completed_games) < games:
                new_games = complete_queue.get()
                completed_games += new_games
                pbar.update(len(new_games))
                game_stat.add_games(new_games)
    for i in range(n_procs):
        procs[i].join()
    complete_queue.close()
    return completed_games, game_stat


@main.command(context_settings={'show_default': True})
@click.argument('model-a', type=click.Path(exists=True, dir_okay=False))
@click.option('--device-a', default='cuda:0', help='empty string for auto')
@click.option('--width-a', type=int, default=8, help='root width for gumbel')
@click.option('--zone-a',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
@click.option('--aux-weight-a', type=click.FloatRange(0, 1), default=0,
              help='weight for auxiliary values in player a'
              )
@click.option(
    '--enable-zone-after-a', type=int, default=0,
    help='ignore zone at opening and exclude from labels in db'
)
@click.argument('model-b', type=click.Path(exists=True, dir_okay=False))
@click.option('--device-b', default='cuda:0', help='empty string for auto')
@click.option('--width-b', type=int, default=8, help='root width for gumbel')
@click.option('--zone-b',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
@click.option('--aux-weight-b', type=click.FloatRange(0, 1), default=0,
              help='weight for auxiliary values in player b'
              )
@click.option(
    '--enable-zone-after-b', type=int, default=0,
    help='ignore zone at opening and exclude from labels in db'
)
@click.option('--games', type=int, help='#games to play',
              default=10)
@click.option('--parallel', type=int, help='#games to play simultaneously',
              default=128)
@click.option('--n-procs', type=int,
              help='#number of process spawned',
              default=8)
@click.option(
    '--output', type=str, default='', help='filename for output'
)
@click.option('--sgf-output', type=click.Path(exists=True, file_okay=False),
              help='folder to store sgfs, ignored if "."',
              default='.')
@click.option(
    '--history-n', type=int, default=7,
    help='history planes for db'
)
@click.option('--tqdm-position', type=int, help='position for tqdm',
              default=0)
@click.option('--tqdm-clear', is_flag=True, help='erase pbar at end')
@click.option('--zone-for-score',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
def match(
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        model_b, width_b, device_b, zone_b, aux_weight_b, enable_zone_after_b,
        games, parallel, n_procs,
        output, sgf_output, history_n,
        tqdm_position, tqdm_clear, zone_for_score
):
    """Play match of MODEL_A (black) v.s. MODEL_B (white).
    """

    completed_games, game_stat = do_match(
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        model_b, width_b, device_b, zone_b, aux_weight_b, enable_zone_after_b,
        games, parallel, n_procs,
        tqdm_position=tqdm_position,
        tqdm_clear=tqdm_clear, zone_for_score=zone_for_score,
    )

    if sgf_output == '.':
        sgf_output = ''
    if output or sgf_output:
        store_completed(
            completed_games, output,
            history_n=history_n,
            ignore_opening_moves=max(enable_zone_after_a, enable_zone_after_b),
            sgf_output=sgf_output,
            zone_black=zone_a, zone_white=zone_b,
        )
    game_stat.report()


def do_evalelo(
        model_a, width_a, device_a, zone_a, aux_weight_a, enable_zone_after_a,
        models, width, device, zone, aux_weight, enable_zone_after,
        games, parallel, n_procs, zone_for_score):
    elos = []
    for model in models:
        _, game_stat_rev = do_match(
            model_a, width_a, device_a, zone_a, aux_weight_a,
            enable_zone_after_a,
            model, width, device, zone, aux_weight, enable_zone_after,
            games, parallel, n_procs, zone_for_score=zone,
        )
        _, game_stat = do_match(
            model, width, device, zone, aux_weight, enable_zone_after,
            model_a, width_a, device_a, zone_a, aux_weight_a,
            enable_zone_after_a,
            games, parallel, n_procs,
            zone_for_score=zone_for_score or zone,
        )
        game_stat.merge(game_stat_rev)
        print(model)
        game_stat.report()
        elos.append(game_stat.black_elo)
    return elos


@main.command(context_settings={'show_default': True})
@click.argument('model-a', type=click.Path(exists=True, dir_okay=False),)
@click.option('--device-a', default='cuda:0', help='empty string for auto')
@click.option('--width-a', type=int, default=8, help='root width for gumbel')
@click.option('--zone-a',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
@click.option('--aux-weight-a', type=click.FloatRange(0, 1), default=0,
              help='weight for auxiliary values in base player'
              )
@click.option(
    '--enable-zone-after-a', type=int, default=0,
    help='ignore zone at opening and exclude from labels in db'
)
@click.argument('models', type=click.Path(exists=True, dir_okay=False),
                nargs=-1)
@click.option('--device', default='cuda:0', help='empty string for auto')
@click.option('--width', type=int, default=8, help='root width for gumbel')
@click.option('--zone',
              type=click.Choice(migo.network.zone_names, case_sensitive=False),
              )
@click.option('--zone2',
              type=click.Choice(migo.network.zone_names, case_sensitive=False),
              )
@click.option('--zone3',
              type=click.Choice(migo.network.zone_names, case_sensitive=False),
              )
@click.option('--zone4',
              type=click.Choice(migo.network.zone_names, case_sensitive=False),
              )
@click.option('--aux-weight', type=click.FloatRange(0, 1), default=0,
              help='weight for auxiliary values in players'
              )
@click.option(
    '--enable-zone-after', type=int, default=0,
    help='ignore zone at opening and exclude from labels in db'
)
@click.option('--games', type=int, help='#games to play',
              default=10)
@click.option('--parallel', type=int, help='#games to play simultaneously',
              default=128)
@click.option('--n-procs', type=int,
              help='#number of process spawned',
              default=8)
@click.option('--zone-for-score',
              type=click.Choice(migo.network.zone_names, case_sensitive=False)
              )
def evalelo(model_a, width_a, device_a, zone_a, aux_weight_a,
            enable_zone_after_a,
            models, width, device, zone, zone2, zone3, zone4, aux_weight,
            enable_zone_after,
            games, parallel, n_procs, zone_for_score):
    """evaluate relative elo of MODELS playing against MODEL_A"""
    results = []
    for i, zone_now in enumerate([zone, zone2, zone3, zone4]):
        if i > 0 and not zone_now:
            continue
        elos = do_evalelo(
            model_a, width_a, device_a, zone_a, aux_weight_a,
            enable_zone_after_a,
            models, width, device, zone_now, aux_weight, enable_zone_after,
            games, parallel, n_procs, zone_for_score
        )
        results.append((zone_now, elos))

    with np.printoptions(formatter={'float': '{: 0.1f}'.format}):
        for zone_now, elos in results:
            print('Zone ', zone_now)
            print(np.array(elos))


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    main()
