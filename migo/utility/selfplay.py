import migo
import migo.dataset
import migo.network
import migo.features
import cygo
import click
import torch
import tqdm
import tqdm.contrib.logging
import numpy as np
import logging


global_config = {}


@click.group()
@click.option('--log-level',
              type=click.Choice(['debug', 'verbose', 'warning', 'quiet'],
                                case_sensitive=False))
def main(log_level):
    """manage migo neural networks"""
    torch.set_float32_matmul_precision('high')

    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    level = logging.WARNING
    match log_level:
        case 'debug': level = logging.DEBUG
        case 'verbose': level = logging.INFO
        case 'warning': level = logging.WARNING
        case 'quiet': level = logging.CRITICAL
    logging.basicConfig(format=FORMAT, level=level, force=True)
    global_config['log_level'] = log_level


class TorchTRTInfer:
    def __init__(self, path: str, device: str):
        import torch_tensorrt
        with torch_tensorrt.logging.info():
            self.trt_module = torch.jit.load(path)
        self.device = device
        print(self.trt_module)

    def infer(self, inputs: torch.Tensor):
        with torch.cuda.device(torch.device(self.device)):
            tensor = inputs.half().to(self.device)
            outputs = self.trt_module(tensor)
            ret = [_.to('cpu').numpy() for _ in outputs]
        return ret


def state_features(state, features: str):
    assert features == 'history8'  # todo
    history_length = 8
    feature_history = migo.features.history_n(state, history_length-1)
    feature_turn = migo.features.color_black(state)
    return np.vstack((feature_history, feature_turn))


class GameManager:
    def __init__(self, board_size, games, parallel):
        self.games = games
        self.parallel = parallel
        self.states = [cygo.State(board_size) for _ in parallel]
        pass


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.option('--device', default='cuda:0', help='empty string for auto')
@click.option('--width', type=int, default=8, help='root width for gumbel')
@click.option('--games', type=int, help='#games to play',
              default=10)
@click.option('--features', type=str, default='history8', help='featureset')
def play(model, width, games, device, features):
    """run selfplay"""
    model = TorchTRTInfer(model, device)
    board_size = 9
    with tqdm.contrib.logging.logging_redirect_tqdm():
        state = cygo.State(board_size)
        input_features = state_features(state, features)
        print(f'{input_features.shape=}')
        ret = model.infer(torch.from_numpy(input_features).unsqueeze(0))
        print(f'{ret[0].shape=} {ret[1].shape=}')


if __name__ == '__main__':
    main()
