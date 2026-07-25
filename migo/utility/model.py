import warnings
import csv
import datetime
import json
import logging
import os
import os.path
import typing

import click
import matplotlib.pyplot as plt
import numpy as np
import recordclass
import torch
import tqdm
import tqdm.contrib.logging

import migo
import migo.cygo as cygo
import migo.dataset
import migo.network

default_csv_name = 'gotrain.csv'

LossStats = recordclass.recordclass(
    'LossStats',
    (
        'move',
        'value',
        'top1',
        'aux_board',
        'aux_value',
    ),
)


def make_loss_stats():
    return LossStats(*np.zeros(len(LossStats())))


def scale_loss_stats(record, scale):
    for i in range(len(record)):
        record[i] *= scale


global_config = {}


@click.group()
@click.option(
    '--log-level',
    type=click.Choice(
        ['debug', 'verbose', 'warning', 'quiet'], case_sensitive=False
    ),
)
def main(log_level):
    """manage migo's neural networks"""
    torch.set_float32_matmul_precision('high')

    FORMAT = '%(asctime)s %(levelname)s %(message)s'
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
    logging.basicConfig(format=FORMAT, level=level, force=True)
    global_config['log_level_str'] = log_level


@main.command(context_settings={'show_default': True})
@click.argument('output', type=click.Path())
@click.option('--board-size', type=int, default=9, help='board size of game')
@click.option('--num-blocks', type=int, default=8, help='block size in network')
@click.option(
    '--channels', type=int, default=128, help='number of channels in network'
)
@click.option('--history-n', type=int, default=7, help='history length')
@click.option(
    '--with-aux-input', is_flag=True, help='add auxiliary plane to input'
)
@click.option('--broadcast-every', type=int, default=3, help='frequency of broadcasting')
@click.option('--initial-weight', type=click.Path(exists=True, dir_okay=False))
def initialize(
    output,
    board_size,
    num_blocks,
    channels,
    history_n,
    with_aux_input,
    broadcast_every,
    initial_weight,
):
    """initialize a model with random weights and save in OUTPUT"""
    extended = with_aux_input
    in_channels = (history_n + 1) * 2 + 1
    optional_args = {}
    if with_aux_input:
        in_channels += 1
        optional_args['with_aux_input'] = True
        optional_args['policy_output_channels'] = 2
    network_class = migo.ExtendedNetwork if extended else migo.PVNetwork
    model = network_class(
        board_size=board_size,
        in_channels=in_channels,
        channels=channels,
        num_blocks=num_blocks,
        broadcast_every=broadcast_every,
        **optional_args,  # ty: ignore[invalid-argument-type]
    )
    if initial_weight:
        model_src, _model_config_src = migo.load_network(initial_weight)
        logging.info(f'loaded {initial_weight}')
        model.load_state_dict(model.state_dict(), strict=False)
    model.save(output)


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
def inspect(model):
    """inspect a MODEL"""
    _model, cfg = migo.load_network(model)
    print(json.dumps(cfg, indent=4))


class Node:
    """node for game tree search"""

    def __init__(self, last_move='', value: float = 0, moves: dict = {}):
        self.value_sum = value
        self.count = 0
        self.moves = moves
        self.children = {}
        self.last_move = last_move

    @property
    def value(self):
        """average value of nodes under subtree"""
        return (self.value_sum / self.count) if self.count else 0

    def az_ucb_score(self, move):
        """return ucb score with AlphaZero's formula"""
        az_pb_c_base = 19652
        az_pb_c_init = 1.25
        pb_c = (
            np.log((self.count + az_pb_c_base + 1) / az_pb_c_base)
            + az_pb_c_init
        )
        child = self.children[move] if move in self.children else None
        count = child.count if child else 0
        pb_c *= np.sqrt(self.count) / (count + 1)
        prior_score = pb_c * float(self.moves[move])
        value_score = (1 - child.value) if child else 0
        return prior_score + value_score

    def select_child_by_ucb(self):
        """select a child to descend in uct"""
        best_score = -1
        best_move = None
        for move in self.children.keys():
            score = self.az_ucb_score(move)
            if best_move is None or best_score < score:
                best_score, best_move = score, move
        if best_move is None:
            for move in self.moves.keys():
                if move in self.children:
                    continue
                score = self.az_ucb_score(move)
                _best_score, best_move = score, move
                self.children[move] = Node(move)  # defer?
                break
        return self.children[best_move]

    def make_tree_dict(self):
        """return dict summarizing subtree"""
        root = self
        data: dict[typing.Any, typing.Any] = {
            'value': f'{root.value:.3f}',
            'count': f'{root.count}',
        }
        for move, prob in root.moves.items():
            if move in self.children:
                child = self.children[move].make_tree_dict()
                data[move] = [prob, child]
            else:
                data[move] = prob
        return data

    def pretty_print(self, indent: int = 4):
        """print subtree"""
        import json

        data = self.make_tree_dict()
        print(json.dumps(data, indent=indent))


def make_input(state: cygo.State, history_n: int):
    """make a batch of length 1 containing input features"""
    import migo.features

    xh = migo.features.history_n(state, history_n)
    xc = migo.features.color(state)
    x = np.vstack((xh, xc))
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def eval_state_by_model(node: Node, model, state: cygo.State):
    """eval state by given model"""
    history_n = (model.config['in_channels'] - 1) // 2 - 1
    x = make_input(state, history_n)
    # print(f'{x.shape=}')

    with torch.no_grad():
        yp, yv = model(x.to(model.device))
    dist = torch.distributions.Categorical(logits=yp).probs[0]  # ty: ignore[not-subscriptable]
    moves = [
        cygo.Move.from_coordinate(*_, state.board_size)
        for _ in state.legal_moves()
    ]
    table = []
    for move in moves:
        idx = move.raw()
        p = dist[idx].item()
        table.append([move.gtp, f'{100 * p:5.2f}'])
    table.append(['pass', f'{100 * dist[81].item():5.2f}'])
    table.sort(key=lambda e: -float(e[1]))
    value = yv[0].item() / 2 + 0.5
    node.value_sum = value
    node.count = 1
    node.moves = dict(table)


def search_one_step(root, model, state):
    """one iteration in mcts"""
    node = root
    path = []
    while len(node.moves) > 0:
        parent = node
        node = parent.select_child_by_ucb()
        move = None
        if node.last_move.lower() != 'pass':
            move = cygo.Move.from_gtp_string(node.last_move, state.board_size)
        state.make_move(move)
        path.append((parent, node.last_move))
    # need batching for practical use
    eval_state_by_model(node, model, state)
    value = 1 - node.value
    while len(path) > 0:
        parent, _ = path.pop()
        parent.value_sum += value
        parent.count += 1
        value = 1 - value


def eval_state_by_ts(model: str | os.PathLike, state, _color, device):
    import torch_tensorrt

    if not device:
        device = 'cuda:0'
    # load from file
    with torch_tensorrt.logging.info():  # ty: ignore[missing-argument, invalid-context-manager]
        trt_module = torch.jit.load(model)
    # inference
    history_n = 7  # todo
    inputs = make_input(state, history_n)  # compose a batch of length 1
    with torch.cuda.device(torch.device(device)):
        tensor = inputs.half().to(device)
        outputs = trt_module(tensor)
        ret = [_.to('cpu').numpy() for _ in outputs]
    print(ret)


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('state', type=click.Path(exists=True, dir_okay=False))
@click.option('--color', type=click.Choice(['black', 'white']), default='black')
@click.option('--device', default='', help='empty string for auto')
@click.option('--budget', type=int, default='0', help='simulate mcts')
def eval_state(model, state, color, device, budget):
    """eval STATE with a MODEL"""
    color = migo.Color.BLACK if color == 'black' else migo.Color.WHITE
    with open(state) as f:
        text = ''.join([_.rstrip() for _ in f])
    state, _ = migo.state.parse(text, next_color=color)
    print(state)
    if model.endswith('ts'):
        eval_state_by_ts(model, state, color, device)
        return

    model, _cfg = migo.load_network(model)
    if not device:
        if torch.cuda.is_available():
            device = 'cuda'
    model = model.to(device)
    model.eval()

    state = state.to_cygo()
    root = Node('')
    eval_state_by_model(root, model, state)
    for _i in range(budget):
        search_one_step(root, model, state.copy())
    root.pretty_print()


def check_consistency(model_config, dataset):
    board_size = dataset.board_size
    in_channels = dataset.input_channels()
    if board_size != model_config['board_size']:
        logging.error(
            f'inconsistency in db and model'
            f' {board_size=} v.s. {model_config["board_size"]}'
        )
        exit(1)
    if in_channels != model_config['in_channels']:
        logging.info(
            f'overwrite history_n in db, as {in_channels=}'
            f' != {model_config["in_channels"]}'
        )
        dataset.history_n = (model_config['in_channels'] - 1) // 2 - 1
        if dataset.input_channels() != model_config['in_channels']:
            logging.error(
                f'{dataset.input_channels()} != {model_config["in_channels"]}'
            )
        assert dataset.input_channels() == model_config['in_channels']


def do_validation(
    model: torch.nn.Module,
    validationloader: torch.utils.data.DataLoader,
    size: int,
    use_pbar: bool,
) -> LossStats:
    criterion = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    running_loss = make_loss_stats()
    model.eval()
    device = model.device

    for i, data in enumerate(
        tqdm.tqdm(
            validationloader,
            total=size,
            disable=not use_pbar,
        )
    ):
        x, yp, yv, *yaux = data
        with torch.no_grad():
            outp, outv, *outaux = model(x.to(device).float())
        yp = yp.to(device).long().squeeze(-1)
        lossp = criterion(outp, yp)
        lossv = mse(outv, yv.to(device).float())
        # loss = lossp + lossv
        top1 = outp.detach().topk(k=1, dim=1)[1]
        top1 = (top1[:, 0] == yp).float().mean()

        running_loss.move += lossp.item()
        running_loss.value += lossv.item()
        running_loss.top1 += top1.item()

        if yaux:
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            loss_aux_p = bce(outaux[0], yaux[0].to(device) / 2 + 0.5)
            loss_aux_v = mse(outaux[1], yaux[1].to(device).float())
            running_loss.aux_board += loss_aux_p
            running_loss.aux_value += loss_aux_v

        if i + 1 >= size:
            break
    scale_loss_stats(running_loss, 1 / size)
    return running_loss


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('testdb', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', type=str, default='', help='empty string for auto')
@click.option(
    '--batch-size', type=int, default=1024, help='batch size for step'
)
@click.option('--size', type=int, default=128, help='#batches')
@click.option(
    '--flip',
    type=click.Choice(['ident', 'udlr', 'ud', 'lr', 'rot90']),
    default='ident',
)
def validate(model, testdb, device, batch_size, size, flip):
    """validate MODEL using TESTDB"""
    dataset = migo.load_dataset(
        testdb,
        batch_with_collate=True,
    )
    dataset.set_transform(flip)
    logging.info(f'load dataset {len(dataset)=} {dataset.input_channels()=}')
    model, model_config = migo.load_network(model)
    check_consistency(model_config, dataset)
    logging.info(f' {dataset.board_size=}')
    validationloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda indices: dataset.collate(indices),
    )
    if not device:
        if torch.cuda.is_available():
            device = 'cuda'
    model = model.to(device)
    vloss = do_validation(
        model,
        validationloader,
        min(size, len(dataset)),
        use_pbar=global_config['log_level_str'] != 'quiet',
    )
    msg = (
        f'validation lossp: {vloss.move:.3f}'
        f' top1: {vloss.top1:.3f}'
        f' lossv: {vloss.value:.3f}'
    )
    if 'with_aux_input' in model_config:
        msg += f' loss_ap: {vloss.aux_board:.3f} loss_av: {vloss.aux_value:.3f}'
    logging.info(msg)


def append_csv(
    csv_path: str | os.PathLike,
    train_loss: LossStats,
    validation_loss: LossStats,
) -> None:
    """
    Appends training and validation loss metrics to a CSV file with a timestamp.

    :param csv_path: The path to the CSV file where data will be appended.
    :param train_loss: LossStats
    :param validation_loss: LossStats
    """
    with open(csv_path, 'a') as csv_output:
        csv_writer = csv.writer(csv_output, quoting=csv.QUOTE_NONNUMERIC)
        now = datetime.datetime.now().isoformat(timespec='seconds')
        csv_writer.writerow(
            [
                now,
                train_loss.move,
                train_loss.value,
                train_loss.top1,
                validation_loss.move,
                validation_loss.value,
                validation_loss.top1,
                train_loss.aux_board,
                train_loss.aux_value,
                validation_loss.aux_board,
                validation_loss.aux_value,
            ]
        )


def try_load_optimizer(
    optimizer: torch.optim.Optimizer, path: str | os.PathLike
) -> bool:
    """try to load optimizer's state_dict from path"""
    objs = torch.load(path)
    if 'optimizer' not in objs:
        return False
    optimizer.load_state_dict(objs['optimizer'])
    return True


def make_pbar_msg(lbl, loss, yaux) -> str:
    msg = (
        f' {lbl}p: {loss.move:.3f}'
        f' {"v" if lbl[0] == "v" else ""}top1: {loss.top1:.3f}'
        f' {lbl}v: {loss.value:.3f}'
    )
    if yaux:
        msg += f' {lbl}_ap: {loss.aux_board:.3f} {lbl}_av: {loss.aux_value:.3f}'
    return msg


def policy_loss(logits, labels, focal_gamma=0, reduction='mean'):
    """generalized cross entropy loss

    c.f. torchvision for binary labels
    https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
    """
    if not focal_gamma or focal_gamma < 0:
        return torch.nn.functional.cross_entropy(logits, labels)
    logpi = torch.nn.functional.log_softmax(logits, dim=-1)
    _action_logpi = logpi.gather(1, labels.unsqueeze(-1))
    action_p = torch.exp(logpi)
    weight = (1 - action_p) ** focal_gamma
    loss = -(logpi * weight)
    if reduction == 'none':
        return loss
    assert reduction == 'mean'
    return loss.mean()


def gce(logits, labels, q=0.5, reduction='mean'):
    """generalized cross entropy loss"""
    p = torch.softmax(logits, dim=-1)
    action_p = p.gather(1, labels.unsqueeze(-1))
    loss = ((1 - action_p) ** q) / q
    return loss

@main.command(context_settings={'show_default': True})
@click.argument(
    'model', type=click.Path(exists=True, dir_okay=False, writable=True)
)
@click.argument(
    'traindb', type=click.Path(exists=True, dir_okay=False), nargs=-1
)
@click.option('--device', type=str, default='', help='empty string for auto')
@click.option(
    '--batch-size', type=int, default=1024, help='batch size for step'
)
@click.option(
    '--batch-limit',
    type=int,
    default=1_000_000,
    # large enough number for default
    help='number of update per epoch',
)
@click.option(
    '--validation-db',
    type=click.Path(exists=True, dir_okay=False),
    help='db for validation',
)
@click.option(
    '--validation-size', type=int, help='size of validation', default=1
)
@click.option(
    '--validation-interval',
    type=int,
    help='frequency of validation',
    default=1000,
)
@click.option(
    '--csv-path',
    type=click.Path(writable=True, dir_okay=False),
    help='path to log stats',
    default=default_csv_name,
)
@click.option(
    '--aux-loss-scale',
    type=float,
    help='weight for auxiliary losses',
    default=0.1,
)
@click.option(
    '--focal-loss', type=float, help='tweak cross entropy if > 0', default=0
)
def train(
    model,
    traindb,
    device,
    batch_size,
    batch_limit,
    validation_db,
    validation_size,
    validation_interval,
    csv_path,
    aux_loss_scale,
    focal_loss,
):
    """train MODEL using TRAINDB"""
    modelpath = model
    output = model
    assert len(traindb) > 0
    dataset = migo.load_dataset(
        traindb[0],
        batch_with_collate=True,
    )
    for dbpath in traindb[1:]:
        db = migo.load_dataset(dbpath, batch_with_collate=True)
        dataset.append(db)
    logging.info(f'load dataset {len(traindb)=} {dataset.input_channels()=}')
    model, model_config = migo.load_network(model)
    check_consistency(model_config, dataset)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda indices: dataset.collate(indices),
    )
    if not device:
        if torch.cuda.is_available():
            device = 'cuda'
    device_is_cuda = device.startswith('cuda')
    model = model.to(device)
    compiled_model = torch.compile(model)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),  # ty: ignore[unresolved-attribute]
        weight_decay=1e-4,
    )
    try_load_optimizer(optimizer, modelpath)
    scaler = torch.amp.GradScaler('cuda', enabled=device_is_cuda)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    validationloader = None
    if validation_db:
        vdataset = migo.load_dataset(
            validation_db,
            batch_with_collate=True,
        )
        check_consistency(model_config, vdataset)
        validationloader = torch.utils.data.DataLoader(
            vdataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda indices: vdataset.collate(indices),
        )
    epoch = 0
    with tqdm.contrib.logging.logging_redirect_tqdm():
        running_loss = make_loss_stats()
        train_iter = iter(trainloader)
        repeat = min(len(trainloader), batch_limit)
        with tqdm.tqdm(
            total=repeat,
            disable=global_config['log_level_str'] == 'quiet',
        ) as pbar:
            pbar.set_description(f'train {modelpath}')
            for i in range(repeat):
                dataset.set_transform(i)
                data = next(train_iter)
                compiled_model.train()  # ty: ignore[unresolved-attribute]
                x, yp, yv, *yaux = data
                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', enabled=device_is_cuda):
                    outp, outv, *outaux = compiled_model(x.to(device).float())

                    yp = yp.to(device).long().squeeze(-1)
                    lossp = policy_loss(outp, yp, focal_loss)
                    lossv = mse(outv, yv.to(device).float())
                    loss = lossp + lossv
                    if yaux:
                        loss_aux_p = bce(
                            outaux[0], yaux[0].to(device) / 2 + 0.5
                        )
                        loss_aux_v = mse(outaux[1], yaux[1].to(device).float())
                        loss += aux_loss_scale * (loss_aux_v + loss_aux_p)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.update(1)

                top1 = outp.detach().topk(k=1, dim=1)[1]
                top1 = (top1[:, 0] == yp).float().mean()
                running_loss.move += lossp.item()
                running_loss.value += lossv.item()
                running_loss.top1 += top1.item()
                if yaux:
                    running_loss.aux_board += loss_aux_p.item()
                    running_loss.aux_value += loss_aux_v.item()

                if i % validation_interval == validation_interval - 1:
                    scale_loss_stats(running_loss, 1 / validation_interval)
                    msg = f'[{epoch + 1},{i + 1:4d}]'
                    msg += make_pbar_msg('loss', running_loss, yaux)
                    if validationloader:
                        vloss = do_validation(
                            compiled_model,  # ty: ignore[invalid-argument-type]
                            validationloader,
                            min(validation_size, len(vdataset)),
                            False,
                        )
                        msg += make_pbar_msg('vloss', vloss, yaux)
                        if csv_path:
                            append_csv(csv_path, running_loss, vloss)
                    logging.info(msg)
                    running_loss = make_loss_stats()

    logging.info('Finished Training')
    torch.save(
        {
            'cfg': model.config,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        output,
    )


@main.command(context_settings={'show_default': True})
@click.argument(
    'csvpath',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    nargs=-1,
)
@click.option('--history-n', type=int, help='history length', default=7)
@click.option(
    '--ignore-opening-moves', type=int, help='number of moves', default=0
)
@click.option('--output', help='path to output', default='./db.npz')
@click.option('--verify', is_flag=True, help='verify moves are legal')
def builddb_from_csv(csvpath, history_n, ignore_opening_moves, output, verify):
    games = []
    for path in csvpath:
        with open(path) as f:
            header = {}
            for lno, line in enumerate(f):
                line = line.strip()
                elems = line.split(',')
                if not header:
                    for i, e in enumerate(elems):
                        header[e.replace('"', '')] = i
                    logging.info(header)
                    continue
                board_size = int(elems[header['boardsize']])
                winner = cygo.Color(int(elems[header['winner']]))
                moves = [int(_) for _ in elems[header['moves']:]]
                raw_moves = np.array(moves, dtype=np.int16)
                games.append(
                    migo.SimpleRecord(
                        board_size=board_size,
                        komi=float(elems[header['komi']]),
                        moves=raw_moves,
                        winner=winner,
                        score=float(elems[header['score']]),
                    )
                )
                if verify:
                    state = cygo.State(
                        board_size, superko_rule=False, max_history_n=0
                    )
                    for mno, move in enumerate(raw_moves):
                        cmove = (
                            cygo.Move.from_raw_value(move, board_size)
                            if move >= 0
                            else cygo.Move.Pass
                        )
                        if not state.is_legal(cmove):
                            print(f'{lno=}, {line=}')
                            print(f'{raw_moves=}')
                            print(state)
                            print(f'{mno=}, {cmove=}, {cmove.gtp=}, {move=}')
                        assert state.is_legal(cmove)
                        state.make_move(cmove)

    dataset = migo.dataset.SgfDataset(
        games=games,
        history_n=history_n,
        ignore_opening_moves=ignore_opening_moves,
    )
    # assume homogeneous games, so OK to see elems yielded the last line
    if ('zone_b' in header and elems[header['zone_b']]
        or 'zone_w' in header and elems[header['zone_w']]):
        zones = np.array(
            [
                migo.network.zone_plane(board_size, elems[header['zone_b']] or 'null'),
                migo.network.zone_plane(board_size, elems[header['zone_w']] or 'null'),
            ]
        )
        dataset = migo.ExtendedDataset.build_from(
            dataset,
            zones,
        )
        
    dataset.save_to(output)


@main.command(context_settings={'show_default': True})
@click.argument('path', type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('--history-n', type=int, help='history length', default=7)
@click.option(
    '--ignore-opening-moves', type=int, help='number of moves', default=0
)
@click.option('--output', help='path to output', default='./db.npz')
def builddb(path, history_n, output, ignore_opening_moves):
    """read sgf games in PATH to store in a single npz file"""
    games = []
    for folder in tqdm.tqdm(path):
        loaded = migo.record.load_sgf_games_in_folder(folder)
        if not loaded:
            logging.info(f'ignored {folder=} with no sgf')
            continue
        games += loaded
        if games[0].board_size != loaded[0].board_size:
            logging.error(
                f'board size mismatch {games[0].board_size}'
                f' != {loaded[0].board_size}'
            )
            exit(1)
    dataset = migo.dataset.SgfDataset(
        games=games,
        history_n=history_n,
        ignore_opening_moves=ignore_opening_moves,
    )
    dataset.save_to(output)


def make_zone_score(area, zone):
    """[-1, 1]"""
    score = np.dot(area.flatten(), zone.flatten())
    score /= zone.sum()
    return score


@main.command(context_settings={'show_default': True})
@click.argument('dbpath', type=click.Path(exists=True, file_okay=True))
@click.option(
    '--zone-black',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option(
    '--zone-white',
    type=click.Choice(migo.network.zone_names, case_sensitive=False),
)
@click.option('--output', help='path to output', default='./dbz.npz')
def buildzonedb(dbpath, zone_black, zone_white, output):
    logging.info(f'loading {dbpath}')
    dataset = migo.SgfDataset.load_from(
        dbpath,
        batch_with_collate=True,
    )
    zones = np.array(
        [
            migo.network.zone_plane(dataset.board_size, zone_black),
            migo.network.zone_plane(dataset.board_size, zone_white),
        ]
    )
    eds = migo.ExtendedDataset.build_from(
        dataset,
        zones,
    )
    logging.info(f'saving to {output}')
    eds.save_to(output)


@main.command(context_settings={'show_default': True})
@click.argument('dbpath', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=True, file_okay=False))
def unpackdb(dbpath, output):
    """read db at DBPATH and extract sgf games into OUTPUT path"""
    dataset = migo.load_dataset(dbpath, batch_with_collate=True)
    for i in range(dataset.n_games()):
        opath = f'{output}/{i}.sgf'
        with open(opath, 'w') as f:
            print(migo.record_to_sgf(dataset.nth_game(i)), file=f)


@main.command()
@click.argument(
    'dbpath', type=click.Path(exists=True, dir_okay=False), nargs=-1
)
def inspectdb(dbpath):
    """show properties of gamedb built by builddb command"""
    import tabulate

    for path in dbpath:
        db = migo.load_dataset(
            path,
            batch_with_collate=True,
        )
        stats = db.summary()
        header = [stats['dbtype'], 'statistics']
        del stats['dbtype']
        lines = [[key, value] for key, value in stats.items()]
        print(
            tabulate.tabulate(
                lines,
                header,
                floatfmt='.3f',
                tablefmt='rst',
            )
        )
        logging.info(f'boardsize {db.board_size}')


@main.command(context_settings={'show_default': True})
@click.option(
    '--csv',
    type=click.Path(exists=True, dir_okay=False),
    help='csv filename',
    default=default_csv_name,
)
@click.option('--output', help='filename of figure', default='./loss.png')
@click.option('--dark', is_flag=True, help='use dark background')
def plot(csv, output, dark):
    """plot a figure of losses recorded in csv"""
    logging.info(f'read {csv=} to output {output}')
    import pandas as pd

    gocsv = pd.read_csv(
        csv,
        names=[
            'date',
            'move',
            'value',
            'top1',
            'vmove',
            'vvalue',
            'vtop1',
            'aux_board',
            'aux_value',
            'vaux_board',
            'vaux_value',
        ],
    )
    _fig, axs = plt.subplots(1, 3, figsize=(9.5, 4))
    if dark:
        plt.style.use('dark_background')

    for ax in axs:
        cid = 'C0'
        ax.xaxis.label.set_color(cid)
        ax.yaxis.label.set_color(cid)
        ax.spines['bottom'].set_color(cid)
        ax.spines['left'].set_color(cid)
        ax.tick_params(axis='x', colors=cid)
        ax.tick_params(axis='y', colors=cid)

    ax = axs[0]
    xlabel = 'positions (x 200Ki)'
    gocsv['top1'].plot(ax=ax, label='train')
    gocsv['vtop1'].plot(ax=ax, label='validation')
    ax.set_ylabel('top1')
    ax.set_xlabel(xlabel)
    # ax.set_ylim(0.4, 0.65)
    ax = axs[1]
    gocsv['value'].plot(ax=ax, label='train')
    gocsv['vvalue'].plot(ax=ax, label='validation')
    ax.set_ylabel('value mse')
    ax.set_xlabel(xlabel)
    # ax.set_ylim(0.25, 0.6)
    ax = axs[2]
    gocsv['move'].plot(ax=ax, label='train')
    gocsv['vmove'].plot(ax=ax, label='validation')
    ax.set_ylabel('policy cross entropy')
    ax.set_xlabel(xlabel)
    # ax.set_ylim(1, 2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output)


def export_onnx(model, in_channels: int, opath, extended: bool):
    import onnx  # to detect import error eaelier

    onnx.__version__
    import torch.onnx

    model.eval()
    dtype = torch.float
    dummy_input = (torch.randn(1024, in_channels, 9, 9, dtype=dtype),)
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'},
    }
    output_names = ['policy', 'value']
    if extended:
        extended_output = ['aux_policy', 'aux_value']
        for name in extended_output:
            dynamic_axes[name] = {0: 'batch_size'}
        output_names += extended_output
    torch.onnx.export(
        model,
        dummy_input,
        opath,
        dynamic_axes=dynamic_axes,
        verbose=False,
        input_names=['input'],
        output_names=output_names,
    )


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', type=str, default='cuda:0', help='cuda:num')
def export_stable(model, device):
    """export MODEL to torch script with TensorRT"""
    base = os.path.splitext(model)[0]
    output, cfg_output = base + '.ts', base + '.json'
    model, cfg = migo.load_network(model)
    in_channels = cfg['in_channels']
    model.eval()
    with open(cfg_output, 'w') as file:
        print(json.dumps(cfg, indent=4), file=file)

    import torch_tensorrt

    model = model.half()
    model = model.to(device)
    # torch_tensorrt.logging.set_is_colored_output_on(True)
    board_size = cfg['board_size']
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, in_channels, board_size, board_size],
            opt_shape=[128, in_channels, board_size, board_size],
            max_shape=[2048, in_channels, board_size, board_size],
            dtype=torch.half,
        )
    ]
    enabled_precisions: set[torch.dtype | torch_tensorrt.dtype] = {torch.half}
    with torch.cuda.device(torch.device(device)):
        input_data = torch.randn(
            16, in_channels, board_size, board_size, device=device, dtype=torch.half
        )
        trt_ts_module = torch_tensorrt.compile(
            torch.jit.script(model),
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            ir='ts',
            # device=torch.device(device)
        )
        _ = trt_ts_module(input_data.half())
    torch.jit.save(trt_ts_module, output)


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', type=str, default='cuda:0', help='cuda:num')
@click.option('--for-python', is_flag=True, help='export pt2 instead of ts')
@click.option('--onnx', is_flag=True, help='export onnx instead of tensorrt')
def export(model, device, for_python, onnx):
    """export MODEL to torch script with TensorRT and new dynamo"""
    warnings.filterwarnings('ignore', 'FutureWarning')
    import torch_tensorrt

    base = os.path.splitext(model)[0]
    output, cfg_output = base + '.ts', base + '.json'
    model, cfg = migo.load_network(model)
    in_channels = cfg['in_channels']
    model.eval()

    if onnx:
        return export_onnx(
            model, in_channels, base + '.onnx',
            'aux_policy_channels' in cfg
        )

    with open(cfg_output, 'w') as file:
        print(json.dumps(cfg, indent=4), file=file)

    model = model.half()
    model = model.to(device)
    board_size = cfg['board_size']
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, in_channels, board_size, board_size],
            opt_shape=[128, in_channels, board_size, board_size],
            max_shape=[2048, in_channels, board_size, board_size],
            dtype=torch.half, device=torch.device(device),
        )
    ]
    # enabled_precisions: set[torch.dtype | torch_tensorrt.dtype] = {torch.half}
    with torch.cuda.device(torch.device(device)):
        input_data = [torch.randn(
            16, in_channels, board_size, board_size, device=device, dtype=torch.half
        )]
        trt_gm = torch_tensorrt.compile(
            model,
            inputs=inputs,
            #enabled_precisions=enabled_precisions,
            ir='dynamo',
            device=torch.device(device),
        )
        if for_python:
            torch_tensorrt.save(trt_gm, f"{base}.pt2", inputs=inputs) # for Python
        else:
            torch_tensorrt.save(trt_gm, output, arg_inputs=input_data, output_format="torchscript")


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--backend', type=str, default='cuda', help='cuda, cpu, or metal')
def export_pte(model, backend):
    '''export for executorch'''
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import executorch.exir
    import executorch.extension

    base = os.path.splitext(model)[0]
    model, cfg = migo.load_network(model)
    model.eval()
    model.half()
    in_channels = cfg['in_channels']
    board_size = cfg['board_size']

    sample_data = (
        torch.randn(
            16,
            in_channels,
            board_size,
            board_size,
            dtype=torch.float16,
        ),
    )
    dynamic_shapes = {
        "x": {
            0: torch.export.Dim("batch", min=1, max=128),
        }
    }

    exported_model = torch.export.export(model, sample_data, dynamic_shapes=dynamic_shapes)
    exported_model = exported_model.run_decompositions(decomp_table=None)
    logging.info('exported')
    if backend == 'coreml':     # torh <= 2.7.0?!
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner

        partitioner = CoreMLPartitioner()
        edge_compile_config = executorch.exir.EdgeCompileConfig()
    elif backend == 'cpu':
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        partitioner = XnnpackPartitioner()
        edge_compile_config = executorch.exir.EdgeCompileConfig()
        
    elif backend == 'cuda':
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        partitioner = CudaPartitioner([
            CudaBackend.generate_method_name_compile_spec("forward"),
            CompileSpec("precision", "fp16".encode('UTF-8')),
            CompileSpec("max_workspace_size", str(1<<33).encode('UTF-8')) # 8GB for autotuning
        ])
        edge_compile_config = executorch.exir.EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        )
        logging.info(f'{edge_compile_config=}')

    logging.info('start edgetransform')
    et_program = executorch.exir.to_edge_transform_and_lower(
        exported_model,
        partitioner=[partitioner],
        compile_config=edge_compile_config,
    )
    logging.info('got et_program')
    exec_program = et_program.to_executorch()
    with open(f'{base}.pte', 'wb') as file:
        exec_program.write_to_file(file)
        # ? executorch.extension.export_util.utils.save_pte_program(exec_program, base, "./output_dir")


@main.command(context_settings={'show_default': True})
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
def verify_pte(model):
    '''verify exported pte model'''
    import executorch.runtime
    runtime = executorch.runtime.Runtime.get()
    # https://docs.pytorch.org/executorch/stable/getting-started.html#testing-the-model
    backends = runtime.backend_registry.registered_backend_names
    print(f"Available backends: {backends}") # cuda?

    program = runtime.load_program(model)
    base = os.path.splitext(model)[0]
    
    with open(base + '.json') as file:
        cfg = json.loads(file.read())
    in_channels = cfg['in_channels']
    board_size = cfg['board_size']
    
    sample_data = (
        torch.randn(
            16,
            in_channels,
            board_size,
            board_size,
        ),
    )
    method = program.load_method("forward")
    assert method
    output = method.execute(sample_data)
    print(f'{len(output)=} {len(output[0])=} {len(output[1])=}')
    print(f'{len(output[0][0])=} {len(output[1][0])=}')


if __name__ == '__main__':
    main()
