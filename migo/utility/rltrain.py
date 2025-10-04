import migo
import torch
import click
import subprocess
import logging
import os
import os.path
import coloredlogs
import tqdm


config = {}


def migo_path(filename):
    import importlib.resources
    return importlib.resources.files('migo') / 'utility' / filename


def call_model(*args, stdout=subprocess.DEVNULL, stderr=None):
    gomodel = str(migo_path('model.py'))
    config['logger'].debug(f"{['python3', gomodel, *args]}")
    ret = subprocess.run(
        ['python3', gomodel, *args],
        stdout=stdout, stderr=stderr
    )
    ret.check_returncode()


def call_selfplay(*args, stdout=subprocess.DEVNULL, stderr=None):
    selfplay = str(migo_path('selfplay.py'))
    config['logger'].debug(f"{['python3', selfplay, *args]}")
    ret = subprocess.run(
        ['python3', selfplay, *args],
        stdout=stdout, stderr=stderr
    )
    ret.check_returncode()


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
    logger = logging.getLogger(__name__)
    FORMAT = '%(asctime)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
    fh = logging.FileHandler("log-auto.txt")
    fh.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    config['logger'] = logger


def train_model(traindb, model, optimizer, device, batch_size, batch_limit,
                *, aux_board_scale, aux_value_scale):
    """train compiled model"""
    primary_value_scale = 1  # 1 - aux_value_scale
    device_is_cuda = device.startswith('cuda')
    dataset = migo.load_dataset(
        traindb[0],
        batch_with_collate=True,
    )
    for dbpath in traindb[1:]:
        db = migo.load_dataset(dbpath, batch_with_collate=True)
        dataset.append(db)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
        collate_fn=lambda indices: dataset.collate(indices),
    )
    with_zone = dataset.aux_input_planes[0].sum() > 0
    center = model.config['board_size']**2+1

    criterion = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device_is_cuda)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    train_iter = iter(trainloader)
    repeat = min(len(trainloader), batch_limit)
    with tqdm.tqdm(total=repeat, leave=False, position=1) as pbar:
        pbar.set_description('train')
        for i in range(repeat):
            dataset.set_transform(i)

            data = next(train_iter)
            model.train()
            x, yp, yv, *yaux = data
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda",
                                enabled=device_is_cuda):
                outp, outv, *outaux = model(x.to(device).float())

                yp = yp.to(device).long().squeeze(-1)
                if with_zone:
                    outp = outp[:, center:]
                else:
                    outp = outp[:, :center]
                lossp = criterion(outp, yp)
                lossv = mse(outv, yv.to(device).float())
                loss = lossp + primary_value_scale * lossv
                if aux_board_scale > 0:
                    loss_aux_p = bce(
                        outaux[0], yaux[0].to(device)/2+0.5
                    )
                    loss += aux_board_scale * loss_aux_p
                if aux_value_scale > 0:
                    loss_aux_v = mse(outaux[1], yaux[1].to(device).float())
                    loss += aux_value_scale * loss_aux_v
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.update(1)

            top1 = outp.detach().topk(k=1, dim=1)[1]
            top1 = (top1[:, 0] == yp).float().mean()


def loop_ages(series, start_age, age_limit, current_model_path, dbs,
              with_zone, games_per_age, games_in_window,
              ignore_opening_moves_for_zone,
              gumbel_root_width, aux_weight,
              n_procs, aux_board_scale
              ):
    from .model import try_load_optimizer
    model, model_config = migo.load_network(current_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    compiled_model = torch.compile(model)
    optimizer = torch.optim.AdamW(compiled_model.parameters(),
                                  weight_decay=1e-4)
    try_load_optimizer(optimizer, current_model_path)
    zone_loop = ['null', 'full', 'center', 'edge']

    with tqdm.trange(age_limit - start_age, colour='#F9E3AA') as pbar:
        for age in range(start_age, age_limit):
            pbar.set_description(f'age {age}')
            zone = None
            if with_zone:
                zone = zone_loop[age % 4]
            # (1) compile into tensorrt
            call_model(
                'export', current_model_path,
                stderr=subprocess.DEVNULL
            )
            basename = os.path.splitext(current_model_path)[0]
            tsmodel = basename + '.ts'
            jsonpath = basename + '.json'
            # (2) self-play
            current_game_path = f'{series}/games{age:04d}'
            ignore_opening_moves = 0
            if zone and zone != 'full' and zone != 'null':
                ignore_opening_moves = ignore_opening_moves_for_zone
            base_cmd = [
                tsmodel,
                '--games', f'{games_per_age}',
                '--output', current_game_path,
                '--n-procs', f'{n_procs}',
                '--enable-zone-after', f'{ignore_opening_moves}',
                '--tqdm-position', '1',
            ]
            if zone:
                cmd = [
                    'play',
                    '--width', f'{gumbel_root_width}',
                    '--zone', f'{zone}',
                    # '--tqdm-clear',
                ]
                if zone != 'null':
                    cmd += ['--aux-weight', f'{aux_weight}']
                cmd += base_cmd
            else:
                cmd = ['play', '--width', f'{gumbel_root_width}'] + base_cmd
            call_selfplay(*cmd)
            dbs.append(current_game_path + '.npz')
            if len(dbs) > games_in_window:
                dbs.pop(0)
            next_model_path = f'{series}/{series}-{age+1:04d}.pth'
            pbar.set_description(f'age {age} train')

            # (3) training
            count = 1024
            if not zone:
                train_model(dbs, compiled_model, optimizer, device,
                            batch_size=1024, batch_limit=count,
                            aux_board_scale=aux_board_scale, aux_value_scale=0)
            elif age < 4:
                for i in range(age+1):
                    train_model(
                        dbs[i::4], compiled_model, optimizer, device,
                        batch_size=1024, batch_limit=count//(age+1),
                        aux_board_scale=aux_board_scale,
                        aux_value_scale=min(1, i)  # skip null zone
                    )
            else:
                for i in range(4):
                    train_model(
                        dbs[i::4], compiled_model, optimizer, device,
                        batch_size=1024, batch_limit=count//4,
                        aux_board_scale=aux_board_scale,
                        aux_value_scale=min(1, i)  # skip null zone
                    )
            torch.save({
                'cfg': model.config,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, next_model_path)

            if age % 10 != 0:
                os.remove(tsmodel)
                os.remove(jsonpath)
            if age % 5 != 0:
                os.remove(current_model_path)
            current_model_path = next_model_path
            pbar.update(1)
    call_model(
        'export', current_model_path,
        stderr=subprocess.DEVNULL
    )


@main.command(context_settings={'show_default': True})
@click.argument('series', default='trial1')
@click.option('--board-size', type=int, default=9,
              help='board size of game')
@click.option('--channels', type=int, default=128,
              help='number of channels in network')
@click.option('--num-blocks', type=int, default=8,
              help='number of blocks in network')
@click.option('--history-n', type=int, default=7,
              help='history length')
@click.option('--with-zone', help='enable zone to focus', is_flag=True)
@click.option('--age-limit', type=int, default=400,
              help='ages to go')
@click.option('--games-per-age', type=int, default=10_000,
              help='games to be added for each age')
@click.option('--games-in-window', type=int, default=1_000_000,
              help='games to be added for each age')
@click.option(
    '--ignore-opening-moves-for-zone', type=int, default=0,
    help='number of moves to be excluded from learning'
)
@click.option('--gumbel-root-width', type=int, default=8,
              help='games to be added for each age')
@click.option('--aux-weight', type=float, default=0.125,
              help='weight of aux value')
@click.option('--n-procs', type=int, default=8,
              help='multiprocessing in playing games')
@click.option('--aux-board-scale', type=float,
              help='weight for auxiliary board loss', default=0.1)
@click.option('--overwrite', is_flag=True, help='overwrite existing run')
def train(
        series,
        board_size, num_blocks, channels, history_n, with_zone,
        age_limit, games_per_age, games_in_window,
        ignore_opening_moves_for_zone,
        gumbel_root_width, aux_weight,
        n_procs, aux_board_scale,
        overwrite
):
    '''MuZero-style reinforcement learning.
    All outputs are stored in SERIES
    '''
    os.makedirs(series, exist_ok=overwrite)
    current_model_path = f'{series}/{series}-0000.pth'
    logging.info(f'build model {current_model_path}')
    cmd = [
        'initialize', current_model_path,
        '--board-size', f'{board_size}',
        '--num-blocks', f'{num_blocks}',
        '--channels', f'{channels}',
        '--history-n', f'{history_n}',
    ]
    if with_zone:
        cmd += ['--with-aux-input']

    call_model(*cmd)
    dbs = []
    loop_ages(series, 0, age_limit, current_model_path, dbs,
              with_zone, games_per_age, games_in_window,
              ignore_opening_moves_for_zone,
              gumbel_root_width, aux_weight,
              n_procs, aux_board_scale
              )


def extract_age(modelname):
    '''extract age from filename

    >>> extract_age('Omachi-0300.pth')
    300
    '''
    age = modelname.replace('-', '.').split('.')[-2]
    return int(age)


@main.command(context_settings={'show_default': True})
@click.argument('checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.option('--with-zone', help='enable zone to focus', is_flag=True)
@click.option('--age-limit', type=int, default=400,
              help='ages to go')
@click.option('--games-per-age', type=int, default=10_000,
              help='games to be added for each age')
@click.option('--games-in-window', type=int, default=1_000_000,
              help='games to be added for each age')
@click.option(
    '--ignore-opening-moves-for-zone', type=int, default=0,
    help='number of moves to be excluded from learning'
)
@click.option('--gumbel-root-width', type=int, default=8,
              help='games to be added for each age')
@click.option('--aux-weight', type=float, default=0.125,
              help='weight of aux value')
@click.option('--n-procs', type=int, default=8,
              help='multiprocessing in playing games')
@click.option('--aux-board-scale', type=float,
              help='weight for auxiliary board loss', default=0.1)
def resume(
        checkpoint, with_zone,
        age_limit, games_per_age, games_in_window,
        ignore_opening_moves_for_zone,
        gumbel_root_width, aux_weight,
        n_procs, aux_board_scale
):
    '''continue learning from checkpoint saved by train.
    '''
    series = os.path.dirname(checkpoint)
    resume_age = extract_age(checkpoint)
    db_size = games_in_window // games_per_age
    dbs = []
    for id in range(max(0, resume_age - db_size), resume_age):
        dbpath = f'{series}/games{id:04d}.npz'
        if not os.path.exists(dbpath):
            raise ValueError(f'db not found {dbpath}')
        dbs.append(dbpath)

    loop_ages(series, resume_age, age_limit, checkpoint, dbs,
              with_zone, games_per_age, games_in_window,
              ignore_opening_moves_for_zone,
              gumbel_root_width, aux_weight,
              n_procs, aux_board_scale
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
