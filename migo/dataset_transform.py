"""transformation for Dataset."""

import numpy as np
import torch


def _is_np(obj) -> bool:
    return isinstance(obj, np.ndarray)


def flip_ident(board_size: int, _board, moveid, _moveplane=None):
    """no rotation or flip, only adjust pass id:  -1 -> board_size^2"""
    if _is_np(moveid):
        ypass = -np.minimum(moveid, 0)
    else:
        ypass = -torch.clamp(moveid, max=0)
    moveid += ypass * (1 + board_size**2)


def flip_lr(board_size: int, board, moveid, moveplane=None):
    """
    flip data horizontary, in place
    :param board_size: H = W
    :param board: tensor with shape (..., H, W)
    :param moveid: array of long, [0, H*W-1] or -1 for pass mapped to H*W
    :param moveplane: tensor with shape (..., H*W+1); board extended with pass
    """
    is_np = _is_np(moveid)
    if is_np:
        board[:] = np.flip(board, axis=-1)
        # 0 or 1
        ypass = -np.minimum(moveid, 0)
    else:
        board[:] = torch.flip(board, dims=[-1])
        ypass = -torch.clamp(moveid, max=0)
        
    moveid += (
        board_size
        - 1
        - (moveid % board_size) * 2
        # -1 -> -board_size -> board_size^2
        + ypass * board_size * (board_size + 1)
    )
    if moveplane is not None:
        focus = moveplane[..., : board_size**2]
        focus = focus.reshape(-1, board_size, board_size)
        if is_np:
            focus[:] = np.flip(focus, axis=-1)
        else:
            focus[:] = torch.flip(focus, dims=[-1])

def flip_udlr(board_size: int, board, moveid, moveplane=None):
    """
    flip data horizontary and vertically, in place
    :param board_size: H = W
    :param board: tensor with shape (..., H, W)
    :param moveid: array of long, [0, H*W-1] or -1 for pass mapped to H*W
    :param moveplane: tensor with shape (..., H*W+1); board extended with pass
    """
    is_np = _is_np(board)
    if is_np:
        board[:] = np.flip(np.flip(board, axis=-2), axis=-1)
    else:
        board[:] = torch.flip(board, dims=[-2, -1])
    # reverse (implicitly including -1 -> board_size^2)
    moveid *= -1
    moveid += board_size**2 - 1
    if moveplane is not None:
        focus = moveplane[..., : board_size**2]
        if is_np:
            focus[:] = np.flip(focus, axis=-1)
        else:
            focus[:] = torch.flip(focus, dims=[-1])


def flip_ud(board_size: int, board, moveid, moveplane=None):
    """
    flip data vertically, in place
    """
    # flip_udlr(board_size, board, moveid, moveplane)
    # flip_lr(board_size, board, moveid, moveplane)
    is_np = _is_np(board)
    move_data = (board_size - 1 - moveid // board_size) * board_size \
            + (moveid % board_size)
    if moveplane is not None:
        focus = moveplane[..., : board_size**2]
    if is_np:
        board[:] = np.flip(board, axis=-2)
        moveid[:] = np.minimum(
            move_data,
            board_size**2,  # -1 -> board_size**2 + board_size - 1
        )
        if moveplane is not None:
            focus[:] = np.flip(
                focus.reshape(-1, board_size, board_size), axis=-2
            ).reshape(-1, board_size**2)
    else:
        board[:] = torch.flip(board, dims=[-2])
        moveid[:] = torch.clamp(
            move_data,
            max=board_size**2,  # -1 -> board_size**2 + board_size - 1
        )
        if moveplane is not None:
            focus[:] = torch.flip(
                focus.reshape(-1, board_size, board_size), dims=[-2]
            ).reshape(-1, board_size**2)


def rot90(board_size: int, board, moveid, moveplane=None):
    """
    rot90
    """
    is_np = _is_np(board)
    if is_np:
        ypass = -np.minimum(moveid, 0)
        board[:] = np.rot90(board, axes=(-2, -1))
    else:
        ypass = -torch.clamp(moveid, max=0)
        board[:] = torch.rot90(board, dims=(-2, -1))
    r, c = moveid // board_size, moveid % board_size
    moveid[:] = r + (board_size - 1 - c) * board_size
    # -1 -> board_size^2
    # moveid += np.maximum(moveid * -(board_size**2 + 1), 0)
    moveid += ypass * (board_size**2 + 1)
    if moveplane is not None:
        focus = moveplane[..., : board_size**2]
        if is_np:
            focus[:] = np.rot90(
                focus.reshape(-1, board_size, board_size), axes=(-2, -1)
            ).reshape(-1, board_size**2)
        else:
            focus[:] = torch.rot90(
                focus.reshape(-1, board_size, board_size), dims=(-2, -1)
            ).reshape(-1, board_size**2)

transforms = [flip_ident, flip_udlr, flip_ud]  # flip_lr, rot90
transforms_dict = {
    'ident': flip_ident,
    'udlr': flip_udlr,
    'ud': flip_ud,
    'lr': flip_lr,
    'rot90': rot90,
}
