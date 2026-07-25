import logging
import os
import os.path
import subprocess

import click
import coloredlogs
import torch
import tqdm

import migo

config = {
    'selfplay_bin': './cygo_cc/build/cygo/utils/cygo-play', # default
    'logger': logging.getLogger(__name__),
}


def logger() -> logging.Logger:
    return config['logger']  # ty: ignore[invalid-return-type]


def selfplay_bin() -> str:
    return config['selfplay_bin']  # ty: ignore[invalid-return-type]


def migo_path(filename):
    import importlib.resources

    return importlib.resources.files('migo') / 'utility' / filename


def call_model(*args, stdout=subprocess.DEVNULL, stderr=None):
    gomodel = str(migo_path('model.py'))
    logger().debug(f'{["python3", gomodel, *args]}')
    ret = subprocess.run(
        ['python3', gomodel, *args], stdout=stdout, stderr=stderr
    )
    ret.check_returncode()


def make_option_name(word: str) -> str:
    return f'--{word.replace("_", "-")}'


def compose_args(arg_dict: dict) -> list[str]:
    ret = [arg_dict['model']]
    if cmd := arg_dict.get('cmd', ''):
        ret = [cmd] + ret
    for k, v in arg_dict.items():
        if k in ['cmd', 'model']:
            continue
        ret += [make_option_name(k), v]
    return ret


def call_selfplay(cmd_dict, stdout=None, stderr=None):
    basecmd = [selfplay_bin()]
    logger().debug(f'{["python3", basecmd, cmd_dict]}')
    if basecmd[0]:
        # user specified cygo-play
        # games are stored in csv
        output_npz = cmd_dict['output']
        if cmd_dict['output'].endswith('.npz'):
            cmd_dict['output'] = cmd_dict['output'][:-4]
        cmd_dict['output'] += '.csv'
        del cmd_dict['cmd']
    else:
        # use selfplay.py
        # games are stored in npz
        selfplay = str(migo_path('selfplay.py'))
        basecmd = ['python3', selfplay]
    ret = subprocess.run(
        basecmd + compose_args(cmd_dict),
        stdout=stdout,
        stderr=subprocess.DEVNULL if basecmd[0] else stderr,
    )
    ret.check_returncode()
    if not basecmd[0].startswith('python'):
        # convert csv to npz
        gomodel = str(migo_path('model.py'))
        cmd = [
            'python3',
            gomodel,
            'builddb-from-csv',
            '--output',
            output_npz,
            cmd_dict['output'],
        ]
        logger().debug(f'{cmd=}')
        ret = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
        )
        ret.check_returncode()
        os.unlink(cmd_dict['output'])


def install_coloredlogs(level: str = 'INFO'):
    fmt = '%(asctime)s %(hostname)s %(levelname)s %(message)s'
    field_styles = {
        'asctime': {'color': 96, 'background': 'white'},
        'hostname': {'color': 112},
        'levelname': {'color': 247},
    }
    coloredlogs.install(level=level, fmt=fmt, field_styles=field_styles)


@click.group()
def main():
    install_coloredlogs()
    # save more to file
    logger_ = logger()
    FORMAT = '%(asctime)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
    fh = logging.FileHandler('log-auto.txt')
    fh.setFormatter(logging.Formatter(FORMAT))
    logger_.addHandler(fh)
    logger_.setLevel(logging.DEBUG)


def make_db_from_path_list(path_list):
    dataset = migo.load_dataset(
        path_list[0],
        batch_with_collate=True,
    )
    # logging.info(f'{len(dataset)=} {path_list[0]}')
    for dbpath in path_list[1:]:
        db = migo.load_dataset(dbpath, batch_with_collate=True)
        # logging.info(f'{len(db)=} {dbpath}')
        dataset.append(db)
    return dataset


def train_model(
    traindb_lst,
    model,
    optimizer,
    device,
    batch_size,
    batch_limit,
    *,
    aux_board_scale,
):
    """train compiled model"""
    from .model import policy_loss

    primary_value_scale = 1  # 1 - aux_value_scale
    device_is_cuda = device.startswith('cuda')
    dataset_lst = [make_db_from_path_list(_) for _ in traindb_lst]
    dataset = migo.dataset.DatasetMixer(dataset_lst)
    transform_in_gpu = True
    if transform_in_gpu:
        dataset.set_transform(None)    # to be done later inside gpu
    board_size = dataset.board_size
    policy_sz = board_size**2 + 1

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda indices: dataset.collate(indices),
    )

    mse = torch.nn.MSELoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda', enabled=device_is_cuda)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    using_mixed_data = not dataset.is_single_db()
    with tqdm.tqdm(total=batch_limit, leave=False, position=1) as pbar:
        pbar.set_description('train')
        model.train()
        for i, data in enumerate(trainloader):
            if i >= batch_limit:
                break
            if not transform_in_gpu:
                dataset.set_transform(i) # set transform for next minibatch
            optimizer.zero_grad()
            x, yp, yv, *yaux = data
            with torch.autocast(device_type='cuda', enabled=device_is_cuda):
                x = x.to(device)
                yp = yp.to(device)
                yv = yv.to(device).float()

                if transform_in_gpu:
                    from migo.dataset_transform import transforms
                    transform_fn = transforms[i % len(transforms)]
                    transform_fn(board_size, x, yp, yaux[0] if yaux else None) # transform in gpu
                yp = yp.long().squeeze(-1) # cast after transform

                outp, outv, *outaux = model(x.float())

                main_policy = outp[:, :policy_sz]  # first half
                sub_policy = outp[:, policy_sz:]
                is_primary = 1
                if using_mixed_data:
                    # data are mixed
                    is_primary = (
                        1
                        - x[:, -1].reshape(-1, board_size**2).max(dim=-1).values
                    )

                # print(f'{main_policy.shape=} {yp.shape=}')
                ce = policy_loss(main_policy, yp, reduction='none')
                lossp = (ce * is_primary).mean()
                if using_mixed_data:
                    lossp_sub = (
                        policy_loss(sub_policy, yp, reduction='none')
                        * (1 - is_primary)
                    ).mean()
                    lossp += lossp_sub
                lossv = mse(outv, yv).mean()
                loss = lossp + primary_value_scale * lossv
                if outaux and aux_board_scale > 0:
                    loss_aux_p = bce(outaux[0], yaux[0].to(device) / 2 + 0.5)
                    loss += aux_board_scale * loss_aux_p
                if using_mixed_data:
                    loss_aux_v = mse(outaux[1], yaux[1].to(device).float()) * (
                        1 - is_primary
                    )
                    loss += loss_aux_v.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.update(1)

            top1 = outp.detach().topk(k=1, dim=1)[1]
            _top1 = (top1[:, 0] == yp).float().mean()


def loop_ages(
    series,
    start_age,
    age_limit,
    current_model_path,
    dbs,
    with_zone,
    games_per_age,
    games_in_window,
    ignore_opening_moves_for_zone,
    gumbel_root_width,
    gumbel_reply_width,
    aux_weight,
    n_procs,
    aux_board_scale,
    device='auto',
    lr: float = 1e-3,
):
    from .model import try_load_optimizer

    model, _model_config = migo.load_network(current_model_path)
    if not device or device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    compiled_model = torch.compile(model)
    # compiled_model = model
    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),  # ty: ignore[unresolved-attribute]
        lr=lr,
        weight_decay=1e-4,
    )
    try_load_optimizer(optimizer, current_model_path)
    zone_loop = ['null', 'full', 'center', 'edge']
    # print('\033[H\033[2J', end='')

    with tqdm.trange(age_limit - start_age, colour='#F9E3AA') as pbar:
        for age in range(start_age, age_limit):
            pbar.set_description(f'{series} age {age}')
            zone = None
            if with_zone:
                zone = zone_loop[age % 4]
            # (1) compile into tensorrt
            call_model('export', current_model_path, stderr=subprocess.DEVNULL)
            basename = os.path.splitext(current_model_path)[0]
            tsmodel = basename + '.ts'
            jsonpath = basename + '.json'
            # (2) self-play
            current_game_path = f'{series}/games{age:04d}'
            ignore_opening_moves = 0
            if zone and zone != 'full' and zone != 'null':
                ignore_opening_moves = ignore_opening_moves_for_zone
            base_cmd_dict = dict(
                model=tsmodel,
                games=f'{games_per_age}',
                output=current_game_path,
                n_procs=f'{n_procs}',
                enable_zone_after=f'{ignore_opening_moves}',
                tqdm_position='1',
                device=device,
                width=f'{gumbel_root_width}',
            )
            if zone:
                cmd_dict = dict(
                    cmd='play',
                    zone=f'{zone}',
                )
                if zone != 'null':
                    cmd_dict['aux_weight'] = f'{aux_weight}'
            else:
                cmd_dict = dict(cmd='play')
            cmd_dict |= base_cmd_dict
            if gumbel_reply_width > 0:
                cmd_dict['reply_width'] = f'{gumbel_reply_width}'
            call_selfplay(cmd_dict)
            dbs.append(current_game_path + '.npz')
            if len(dbs) * games_per_age > games_in_window:
                if with_zone:
                    del dbs[:4]
                else:
                    dbs.pop(0)
            # print('\033[H\033[K', end='')
            next_model_path = f'{series}/{series}-{age + 1:04d}.pth'
            pbar.set_description(f'{series} age {age}') # redraw

            # (3) training
            count = 1024
            if not zone:
                train_model(
                    [dbs],
                    compiled_model,
                    optimizer,
                    device,
                    batch_size=1024,
                    batch_limit=count,
                    aux_board_scale=aux_board_scale,
                )
            else:
                length = min(4, age + 1)
                db_lst = [dbs[i::4] for i in range(length)]
                train_model(
                    db_lst,
                    compiled_model,
                    optimizer,
                    device,
                    batch_size=1024,
                    batch_limit=count,
                    aux_board_scale=aux_board_scale,
                )
            torch.save(
                {
                    'cfg': model.config,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                next_model_path,
            )

            if age % 10 != 0:
                os.remove(tsmodel)
                os.remove(jsonpath)
            if age % 5 != 0:
                os.remove(current_model_path)
            current_model_path = next_model_path
            pbar.update(1)
    call_model('export', current_model_path, stderr=subprocess.DEVNULL)


@main.command(context_settings={'show_default': True})
@click.argument('series', default='trial1')
@click.option('--board-size', type=int, default=9, help='board size of game')
@click.option(
    '--channels', type=int, default=128, help='number of channels in network'
)
@click.option(
    '--num-blocks', type=int, default=8, help='number of blocks in network'
)
@click.option('--history-n', type=int, default=7, help='history length')
@click.option('--with-zone', help='enable zone to focus', is_flag=True)
@click.option('--broadcast-every', type=int, default=3, help='frequency of broadcasting')
@click.option('--initial-weight', type=click.Path(exists=True, dir_okay=False))
@click.option('--age-limit', type=int, default=400, help='ages to go')
@click.option(
    '--games-per-age',
    type=int,
    default=10_000,
    help='games to be added for each age',
)
@click.option(
    '--games-in-window',
    type=int,
    default=1_000_000,
    help='games to be added for each age',
)
@click.option(
    '--ignore-opening-moves-for-zone',
    type=int,
    default=0,
    help='number of moves to be excluded from learning',
)
@click.option(
    '--gumbel-root-width',
    type=int,
    default=8,
    help='number of moves compared at each decision',
)
@click.option(
    '--gumbel-reply-width',
    type=int,
    default=0,
    help='number of replies considered at each decision',
)
@click.option(
    '--aux-weight', type=float, default=0.125, help='weight of aux value'
)
@click.option(
    '--n-procs', type=int, default=8, help='multiprocessing in playing games'
)
@click.option(
    '--aux-board-scale',
    type=float,
    help='weight for auxiliary board loss',
    default=0.0,
)
@click.option(
    '--lr',
    type=float,
    help='learning rate',
    default=1e-3,
)
@click.option('--overwrite', is_flag=True, help='overwrite existing run')
@click.option('--device', default='auto', help='cpu or cuda')
@click.option(
    '--selfplay-bin',
    default=selfplay_bin(),
    help='path to cygo-play or empty string for selfplay.py',
)
def train(
    series,
    board_size,
    num_blocks,
    channels,
    history_n,
    with_zone,
    broadcast_every,
    initial_weight,
    age_limit,
    games_per_age,
    games_in_window,
    ignore_opening_moves_for_zone,
    gumbel_root_width,
    gumbel_reply_width,
    aux_weight,
    n_procs,
    aux_board_scale,
    overwrite,
    device,
    selfplay_bin,
    lr
):
    """MuZero-style reinforcement learning.
    All outputs are stored in SERIES
    """
    os.makedirs(series, exist_ok=overwrite)
    current_model_path = f'{series}/{series}-0000.pth'
    logging.info(f'build model {current_model_path}')
    cmd = [
        'initialize',
        current_model_path,
        '--board-size',
        f'{board_size}',
        '--num-blocks',
        f'{num_blocks}',
        '--channels',
        f'{channels}',
        '--history-n',
        f'{history_n}',
        '--broadcast-every', f'{broadcast_every}'
    ]
    if with_zone:
        cmd += ['--with-aux-input']
    if initial_weight:
        cmd += ['--initial-weight', f'{initial_weight}']

    call_model(*cmd)

    if not os.path.exists(selfplay_bin):
        if selfplay_bin not in ['',  config['selfplay_bin']]:
            logging.error(f'cannot open {selfplay_bin=}')
            exit(1)
        if selfplay_bin != '':
            logging.warning(f'cannot access {selfplay_bin=}, fallback to Python')
        else:
            logging.info(f'use selfplay.py')
        selfplay_bin = ''

    config['selfplay_bin'] = selfplay_bin

    dbs = []
    loop_ages(
        series,
        0,
        age_limit,
        current_model_path,
        dbs,
        with_zone,
        games_per_age,
        games_in_window,
        ignore_opening_moves_for_zone,
        gumbel_root_width,
        gumbel_reply_width,
        aux_weight,
        n_procs,
        aux_board_scale,
        device,
        lr=lr
    )


def extract_age(modelname):
    """extract age from filename

    >>> extract_age('Omachi-0300.pth')
    300
    """
    age = modelname.replace('-', '.').split('.')[-2]
    return int(age)


@main.command(context_settings={'show_default': True})
@click.argument('checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.option('--with-zone', help='enable zone to focus', is_flag=True)
@click.option('--age-limit', type=int, default=400, help='ages to go')
@click.option(
    '--games-per-age',
    type=int,
    default=10_000,
    help='games to be added for each age',
)
@click.option(
    '--games-in-window',
    type=int,
    default=1_000_000,
    help='games to be added for each age',
)
@click.option(
    '--ignore-opening-moves-for-zone',
    type=int,
    default=0,
    help='number of moves to be excluded from learning',
)
@click.option(
    '--gumbel-root-width',
    type=int,
    default=8,
    help='games to be added for each age',
)
@click.option(
    '--gumbel-reply-width',
    type=int,
    default=0,
    help='number of replies considered at each decision',
)
@click.option(
    '--aux-weight', type=float, default=0.125, help='weight of aux value'
)
@click.option(
    '--n-procs', type=int, default=8, help='multiprocessing in playing games'
)
@click.option(
    '--aux-board-scale',
    type=float,
    help='weight for auxiliary board loss',
    default=0.0,
)
@click.option('--device', default='auto', help='cpu or cuda')
@click.option(
    '--selfplay-bin',
    default=selfplay_bin(),
    help='path to cygo-play or empty string for selfplay.py',
)
def resume(
    checkpoint,
    with_zone,
    age_limit,
    games_per_age,
    games_in_window,
    ignore_opening_moves_for_zone,
    gumbel_root_width,
    gumbel_reply_width,
    aux_weight,
    n_procs,
    aux_board_scale,
    device,
    selfplay_bin,
):
    """continue learning from checkpoint saved by train."""
    if checkpoint.endswith('.ts'):
        logging.error(f'expected filename.pth but received {checkpoint}')
        return 1
    series = os.path.dirname(checkpoint)
    resume_age = extract_age(checkpoint)
    db_size = games_in_window // games_per_age
    dbs = []
    for id in range(max(0, resume_age - db_size), resume_age):
        dbpath = f'{series}/games{id:04d}.npz'
        if not os.path.exists(dbpath):
            raise ValueError(f'db not found {dbpath}')
        dbs.append(dbpath)

    config['selfplay_bin'] = selfplay_bin

    loop_ages(
        series,
        resume_age,
        age_limit,
        checkpoint,
        dbs,
        with_zone,
        games_per_age,
        games_in_window,
        ignore_opening_moves_for_zone,
        gumbel_root_width,
        gumbel_reply_width,
        aux_weight,
        n_procs,
        aux_board_scale,
        device,
    )


@main.command()
def selfcheck():
    gomodel = migo_path('model.py')
    logging.info(gomodel)
    call_model('--help')
    selfplay = migo_path('selfplay.py')
    logging.info(selfplay)
    call_selfplay('--help')
    logging.info('success')


if __name__ == '__main__':
    main()
