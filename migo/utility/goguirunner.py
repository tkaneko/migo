import migo
import migo.gtp
import cygo
import numpy as np
import click
import sys
import io
import os.path
import json


def np_softmax(policy):
    policy = policy - policy.max()
    exp = np.exp(policy)
    return exp / exp.sum()


def np_sigmoid(x):
    exp = np.exp(-x)
    return 1 / exp


class MigoGoGuiRunner(migo.gtp.GoGuiGTPRunner):
    def __init__(self, board_size):
        super().__init__()
        self.ort_session = None
        self.gtp_params = migo.gtp.GoGuiParams([
            migo.gtp.GoGuiParam('boardsize', int, 'int', board_size),
            migo.gtp.GoGuiParam('komi', float, 'float', 7.0)
        ])

        self.add_callback('boardsize', self.set_boardsize, arity=1)
        self.add_callback('komi', self.set_komi, arity=1)
        self.add_callback('clear_board', self.clear_board, arity=0)
        self.add_callback('showboard', self.showboard, arity=0)
        self.add_callback('play', self.gtp_play, arity=2)
        self.add_callback('name', self.name, arity=0)
        self.add_callback('version', self.version, arity=0)
        self.add_callback('load_eval', self.load_eval, arity=1)
        self.add_callback('genmove', self.genmove, arity=1)

        self.add_analyze_callback(
            migo.gtp.CommandType.BWBOARD,
            'tromp_taylor', self.tromp_taylor_bw
        )
        self.add_analyze_callback(
            migo.gtp.CommandType.CBOARD,
            'view_policy', self.view_policy
        )
        self.add_analyze_callback(
            migo.gtp.CommandType.CBOARD,
            'predict_territory', self.predict_territory
        )

        self.clear_board()

    def name(self, *args):
        return migo.gtp.Status.success, 'migo'

    def version(self, *args):
        return migo.gtp.Status.success, migo.version()

    def set_boardsize(self, *args):
        self.gtp_params.boardsize = int(args[0])
        return migo.gtp.Status.success, self.gtp_params.boardsize

    def set_komi(self, *args):
        self.gtp_params.komi = float(args[0])
        return migo.gtp.Status.success, self.gtp_params.komi

    def clear_board(self, *args):
        self.game = cygo.State(
            board_size=self.gtp_params.boardsize,
            komi=self.gtp_params.komi
        )
        return migo.gtp.Status.success, (
            self.gtp_params.boardsize,
            self.gtp_params.komi
        )

    def showboard(self, *args):
        with io.StringIO() as output:
            print(self.game, file=output)
            return migo.gtp.Status.success, output.getvalue()

    def gtp_play(self, *args):
        color = int(migo.gtp.parse_color(args[0]))
        action = None
        if args[1] != "PASS":
            move = migo.gtp.parse_move(args[1])
            row = move[0]
            col = move[1]
            if move is not None:
                action = self.game.board_size * row + col
        self.game.make_move(action, color=cygo.Color(color))
        return migo.gtp.Status.success, args[1]

    def tromp_taylor_bw(self, *args):
        tt = self.game.tromp_taylor_fill().tolist()
        bwboard = ['']
        lbls = '-BW'  # for 0, 1, -1
        for row in reversed(tt):
            line = [lbls[sq] for sq in row]
            bwboard.append(' '.join(line))
        return migo.gtp.Status.success, '\n'.join(bwboard)

    def make_normalized_color(self, policy):
        import matplotlib.colors
        cmap = matplotlib.colormaps['plasma']
        min_value = policy.min()
        max_value = max(policy.max(), min_value + 1)
        if max_value - min_value > 1.0:
            # [0, 1.0]
            policy = (policy - min_value) / (max_value - min_value) * 1.0
        return [
            matplotlib.colors.to_hex(cmap(_))
            for _ in policy
        ]

    def infer(self):
        history_n = (self.eval_config['in_channels'] - 1) // 2 - 1
        if 'aux_policy_channels' in self.eval_config:
            board_size = self.game.board_size
            zone = np.ones((board_size, board_size), dtype=np.int8)
            x, _ = cygo.features.batch_features_with_zone(
                [self.game],
                history_n, zone
            )
        else:
            x, _ = cygo.features.batch_features(
                [self.game],        # batch of size 1
                history_n
            )
        out = self.ort_session.run(None, {"input": x.astype(np.float32)})
        return [_[0] for _ in out]

    def infer_policy(self):
        out = self.infer()
        out = out[0]  # (policy, value, ...)
        return np_softmax(out)

    def infer_results(self):
        out = self.infer()
        prediction = out[2]  # (policy, value, aux_p, aux_v)
        return np_sigmoid(prediction)

    def view_policy(self, *args):
        if not self.ort_session:
            return migo.gtp.Status.failure, 'onnx not loaded yet'
        board_size = self.game.board_size
        lines = []
        policy = self.infer_policy()[:board_size**2]
        # exaggerate temperature by sqrt
        policy = np.sqrt(policy)
        cboard = self.make_normalized_color(policy)
        for y in reversed(range(board_size)):
            line = ' '.join(cboard[y*board_size:(y+1)*board_size])
            lines.append(line)
        return migo.gtp.Status.success, '\n'.join(lines)

    def predict_territory(self, *args):
        if not self.ort_session:
            return migo.gtp.Status.failure, 'onnx not loaded yet'
        if 'aux_policy_channels' not in self.eval_config:
            return migo.gtp.Status.failure, 'not supported in current eval'
        board_size = self.game.board_size
        lines = []
        policy = self.infer_results()[:board_size**2]
        cboard = self.make_normalized_color(policy)
        for y in reversed(range(board_size)):
            line = ' '.join(cboard[y*board_size:(y+1)*board_size])
            lines.append(line)
        return migo.gtp.Status.success, '\n'.join(lines)

    def load_eval(self, *args):
        path = args[0]
        if not os.path.exists(path):
            return migo.gtp.Status.failure, 'file not found'
        cfg_path = os.path.splitext(path)[0] + '.json'
        if not os.path.exists(cfg_path):
            return migo.gtp.Status.failure, f'cfg file not found {cfg_path}'
        with open(cfg_path) as file:
            self.eval_config = json.loads(file.read())
        import onnxruntime as ort
        provider = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(path, providers=provider)
        in_channels = self.eval_config["in_channels"]
        return migo.gtp.Status.success, f'{path} {in_channels=}'

    def genmove(self, *args):
        if not self.ort_session:
            return migo.gtp.Status.failure, 'onnx not loaded yet'
        color = args[0]
        color = cygo.BLACK if color == 'b' else cygo.WHITE
        policy = self.infer_policy()
        move_id = policy.argmax()
        if move_id < self.game.board_size**2:
            move = cygo.Move.from_raw_value(move_id, self.game.board_size)
        else:
            move = None
        self.game.make_move(move, color)
        return migo.gtp.Status.success, (move.gtp if move else 'PASS')


@click.command
@click.option('--board-size', type=int, default=9)
@click.option('--load-eval', type=click.Path(readable=True, dir_okay=False),
              default='/dev/null')
def main(board_size, load_eval):
    migo_runner = MigoGoGuiRunner(board_size)
    if load_eval != '/dev/null':
        ret, msg = migo_runner.load_eval(load_eval)
        print(f'load eval {ret=} {msg}', file=sys.stderr)
    migo_runner.execute()


if __name__ == '__main__':
    main()
