import numpy as np

from migo import State, Color
from .utils import Order, concatenate_along, empty_planes, \
    one_planes, zero_planes


def board(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    planes = zero_planes(2, state.board_size, order, dtype=dtype)

    if order == Order.CHW:
        planes[0, :, :] = state.board == state.current_player
        planes[1, :, :] = state.board == state.current_player.opponent()

    else:
        planes[:, :, 0] = state.board == state.current_player
        planes[:, :, 1] = state.board == state.current_player.opponent()

    return planes


def board_i(state: State, i: int, order=Order.CHW, dtype=np.float32
            ) -> np.ndarray:
    assert i <= state.max_history_n

    if i == 0:
        return board(state, order, dtype)

    if i > len(state.history_buffer):
        return zero_planes(2, state.board_size, order, dtype=dtype)

    planes = zero_planes(2, state.board_size, order, dtype=dtype)
    history_board = state.history_buffer[-i]

    if order == Order.CHW:
        planes[0, :, :] = history_board == state.current_player
        planes[1, :, :] = history_board == state.current_player.opponent()
    else:
        planes[:, :, 0] = history_board == state.current_player
        planes[:, :, 1] = history_board == state.current_player.opponent()

    return planes


def history_n(state: State, n: int, order=Order.CHW, dtype=np.float32
              ) -> np.ndarray:
    assert n >= 0

    boards = [board_i(state, i, order, dtype) for i in range(n+1)]

    return concatenate_along(order, boards)


def color(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    return color_black(state, order, dtype)


def color_black(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    return one_planes(1, state.board_size, order, dtype) \
        * (state.current_player == Color.BLACK)


def color_white(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    return one_planes(1, state.board_size, order, dtype) \
        * (state.current_player == Color.WHITE)


def stone_color(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((3, state.board_size, state.board_size), dtype=dtype)

        planes[0, :, :] = state.board == state.current_player
        planes[1, :, :] = state.board == state.current_player.opponent()
        planes[2, :, :] = state.board == Color.EMPTY

    else:
        planes = np.zeros((state.board_size, state.board_size, 3),
                          dtype=np.float32)

        planes[:, :, 0] = state.board == state.current_player
        planes[:, :, 1] = state.board == state.current_player.opponent()
        planes[:, :, 2] = state.board == Color.EMPTY

    return planes


def turns_since(state: State, order=Order.CHW, nb_planes=8, dtype=np.float32
                ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((nb_planes, state.board_size, state.board_size),
                          dtype=dtype)

        for i in range(nb_planes - 1):
            planes[i, state.stone_ages == i] = 1

        planes[nb_planes - 1, state.stone_ages >= nb_planes - 1] = 1

    else:
        planes = np.zeros((state.board_size, state.board_size, nb_planes),
                          dtype=dtype)

        for i in range(nb_planes - 1):
            planes[state.stone_ages == i, i] = 1

        planes[state.stone_ages >= nb_planes - 1, nb_planes - 1] = 1

    return planes


def liberties(state: State, order=Order.CHW, nb_planes=8, dtype=np.float32
              ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((nb_planes, state.board_size, state.board_size),
                          dtype=dtype)

        for i in range(nb_planes):
            planes[i, state.liberty_counts == i + 1] = 1

        planes[nb_planes - 1, state.liberty_counts >= nb_planes] = 1

    else:
        planes = np.zeros((state.board_size, state.board_size, nb_planes),
                          dtype=dtype)

        for i in range(nb_planes):
            planes[state.liberty_counts == i + 1, i] = 1

        planes[state.liberty_counts >= nb_planes, nb_planes - 1] = 1

    return planes


def capture_size(state: State, order=Order.CHW, nb_planes=8, dtype=np.float32
                 ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((nb_planes, state.board_size, state.board_size),
                          dtype=dtype)
    else:
        planes = np.zeros((state.board_size, state.board_size, nb_planes),
                          dtype=dtype)

    for x, y in state.legal_moves():
        nb_captured = 0

        for neighbors in state.groups_around_at((x, y)):
            neighbor = next(iter(neighbors))  # get an arbitrary stone in the 'neighbors'

            if state.liberty_counts[neighbor] == 1 \
               and state.board[neighbor] != state.current_player:
                # if the group has 1 liberty and the group is not my color,
                # the group would be captured
                nb_captured += len(state.groups[neighbor])

        if order == Order.CHW:
            planes[min(nb_captured, nb_planes - 1), x, y] = 1
        else:
            planes[x, y, min(nb_captured, nb_planes - 1)] = 1

    return planes


def self_atari_size(state: State, order=Order.CHW, nb_planes=8, dtype=np.float32
                    ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((nb_planes, state.board_size, state.board_size),
                          dtype=dtype)
    else:
        planes = np.zeros((state.board_size, state.board_size, nb_planes),
                          dtype=dtype)

    for x, y in state.legal_moves():
        liberty_set_after = set(state.liberty_sets[(x, y)])
        group_after_move = {(x, y)}
        captured_stones = set()

        for neighbors in state.groups_around_at((x, y)):
            neighbor = next(iter(neighbors))  # get an arbitrary stone in 'neighbors'

            if state.board[neighbor] == state.current_player:
                # if the group is mine, take them into account
                liberty_set_after |= state.liberty_sets[neighbor]
                group_after_move |= state.groups[neighbor]

            elif state.liberty_counts[neighbor] == 1:
                # if the group is enemy's and the group would be captured
                # (neighbor cannot be Color.EMPTY because neighbors has one liberty)
                captured_stones |= state.groups[neighbor]

        if captured_stones:
            for stone in group_after_move:
                # if there are some groups that can be captured,
                # the coordinates in which captured group was in can be new liberties of mine
                liberty_set_after |= (set(state.crosswise_neighbors_of(stone))
                                      & captured_stones)

        if (x, y) in liberty_set_after:
            liberty_set_after.remove((x, y))

        if len(liberty_set_after) == 1:
            if order == Order.CHW:
                planes[min(nb_planes - 1, len(group_after_move) - 1), x, y] = 1
            else:
                planes[x, y, min(nb_planes - 1, len(group_after_move) - 1)] = 1

    return planes


def liberties_after_move(state: State, order=Order.CHW, nb_planes=8,
                         dtype=np.float32) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((nb_planes, state.board_size, state.board_size),
                          dtype=dtype)
    else:
        planes = np.zeros((state.board_size, state.board_size, nb_planes),
                          dtype=dtype)

    for x, y in state.legal_moves():
        liberty_set_after = set(state.liberty_sets[(x, y)])
        group_after = {(x, y)}
        captured_stones = set()

        for neighbors in state.groups_around_at((x, y)):
            neighbor = next(iter(neighbors))

            if state.board[neighbor] == state.current_player:
                liberty_set_after |= state.liberty_sets[neighbor]
                group_after |= state.groups[neighbor]

            elif state.liberty_counts[neighbor] == 1:
                captured_stones |= state.groups[neighbor]

        if captured_stones:
            for stone in group_after:
                liberty_set_after |= (set(state.crosswise_neighbors_of(stone))
                                      & captured_stones)

        if (x, y) in liberty_set_after:
            liberty_set_after.remove((x, y))

        if order == Order.CHW:
            planes[min(nb_planes - 1, len(liberty_set_after) - 1), x, y] = 1
        else:
            planes[x, y, min(nb_planes - 1, len(liberty_set_after) - 1)] = 1

    return planes


def ladder_capture(state: State, order=Order.CHW, dtype=np.float32
                   ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((1, state.board_size, state.board_size), dtype=dtype)

        for x, y in state.legal_moves():
            planes[0, x, y] = state.is_ladder_escape((x, y))
    else:
        planes = np.zeros((state.board_size, state.board_size, 1), dtype=dtype)

        for x, y in state.legal_moves():
            planes[x, y, 0] = state.is_ladder_escape((x, y))

    return planes


def ladder_escape(state: State, order=Order.CHW, dtype=np.float32
                  ) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((1, state.board_size, state.board_size), dtype=dtype)

        for x, y in state.legal_moves():
            planes[0, x, y] = state.is_ladder_capture((x, y))
    else:
        planes = np.zeros((state.board_size, state.board_size, 1), dtype=dtype)

        for x, y in state.legal_moves():
            planes[x, y, 0] = state.is_ladder_capture((x, y))

    return planes


def sensibleness(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    if order == Order.CHW:
        planes = np.zeros((1, state.board_size, state.board_size), dtype=dtype)

        for x, y in state.legal_moves(include_eyes=False):
            planes[0, x, y] = 1

    else:
        planes = np.zeros((state.board_size, state.board_size, 1), dtype=dtype)

        for x, y in state.legal_moves(include_eyes=False):
            planes[x, y, 0] = 1

    return planes


def leela_board(state: State, order=Order.CHW, n=8, dtype=np.float32
                ) -> np.ndarray:
    assert n <= state.max_history_n

    def write(j, src, dst):
        k = n + j
        if order == Order.CHW:
            dst[j, :, :] = src[0, :, :]
            dst[k, :, :] = src[1, :, :]
        else:
            dst[:, :, j] = src[:, :, 0]
            dst[:, :, k] = src[:, :, 1]

    planes = empty_planes(2 * n, state.board_size, order, dtype)

    for i in range(n):
        write(i, board_i(state, i, order, dtype=dtype), planes)

    return planes


def leela_color(state: State, order=Order.CHW, dtype=np.float32) -> np.ndarray:
    if order == Order.CHW:
        planes = np.empty((2, state.board_size, state.board_size), dtype=dtype)

        planes[0, :, :] = one_planes(1, state.board_size, order, dtype) \
            * (state.current_player == Color.BLACK)
        planes[1, :, :] = one_planes(1, state.board_size, order, dtype) \
            * (state.current_player == Color.WHITE)

    else:
        planes = np.empty((state.board_size, state.board_size, 2), dtype=dtype)

        planes[:, :, 0] = one_planes(1, state.board_size, order, dtype) \
            * (state.current_player == Color.BLACK)
        planes[:, :, 1] = one_planes(1, state.board_size, order, dtype) \
            * (state.current_player == Color.WHITE)

    return planes
