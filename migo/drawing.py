"""draw board image as matplotlib figs
"""
from migo import Color, State, all_coordinates
import migo.features
import cygo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


default_config = {
    "fig": {9: 7, 13: 8, 19: 8},
    "font": {9: 18, 13: 12, 19: 12},
    "marker": {9: 36, 19: 20},
    "center": {9: 10, 13: 5, 19: 5},
    'facecolor': (1, 1, .8),
    'margin': {9: 0.12, 19: 0.16},
}


def make_key(board_size: int) -> int:
    return max(9, min(19, board_size))


def make_fig(board_size: int = 9,
             config: dict = {},
             with_note: bool = False):
    """make plain figure

    >>> fig, ax, ax_note = migo.drawing.make_fig(9)
    >>> assert fig is not None
    >>> assert ax is not None
    >>> assert ax_note is None
    >>> fig, ax, ax_note = migo.drawing.make_fig(9, with_note=True)
    >>> assert ax is not None
    >>> assert ax_note is not None
    """
    config = default_config | config
    board_key = make_key(board_size)
    figsize = config["fig"][board_key]

    if with_note:
        fig = plt.figure(figsize=[figsize*1.1, figsize])
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[5, 1])
        ax = fig.add_subplot(gs[0])
        ax_notes = fig.add_subplot(gs[1])
        ax_notes.axis('off')
    else:
        fig = plt.figure(figsize=[figsize, (figsize - 0.5) * 6 / 5])
        ax = fig.add_subplot(111)
        ax_notes = None
    fig.subplots_adjust(left=0.01, right=1, bottom=0.01, top=1)
    setup_board_ax(board_size, fig, ax, config)
    return fig, ax, ax_notes


def ax_scale(board_size, fig, ax):
    """return scale of ax"""
    board_key = make_key(board_size)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    return (width / default_config["fig"][board_key]) ** 0.8


def setup_board_ax(board_size, fig, ax, config: dict = {}):
    """draw empty board"""
    import string
    config = default_config | config
    board_key = make_key(board_size)
    scale = ax_scale(board_size, fig, ax)
    fontsize = config["font"][board_key] * scale
    facecolor = config["facecolor"]

    fig.patch.set_facecolor(facecolor)
    ax.set_axis_off()
    for x in range(board_size):
        ax.plot([x, x], [0, board_size - 1], 'k')
    for y in range(board_size):
        ax.plot([0, board_size - 1], [y, y], 'k')

    ax.set_xlim(-1, board_size)
    ax.set_ylim(-1, board_size)
    y_labels = list(string.ascii_uppercase[:board_size+1].replace("I", ""))
    x_labels = [str(i) for i in range(1, board_size+1)]

    for w in range(board_size):
        ax.text(-0.6, w, x_labels[w], va='center', ha='right',
                fontsize=fontsize, color='black')

    for h in range(board_size):
        ax.text(h, board_size-0.6, y_labels[h], va='bottom', ha='center',
                fontsize=fontsize, color='black')

    if board_size % 2:
        markersize = config['center'][board_key] * scale
        ax.plot(
            (board_size - 1) // 2, (board_size - 1) // 2, 'o',
            markersize=markersize,
            markeredgecolor='k', markerfacecolor='k', markeredgewidth=2
        )

    if board_size >= 13:
        coords = (
            # corner
            (4 - 1, 4 - 1), (board_size - 4, board_size - 4),
            (4 - 1, board_size - 4), (board_size - 4, 4 - 1),
            # etdge
            ((board_size - 1) // 2, 4 - 1),
            ((board_size - 1) // 2, board_size - 4),
            (4 - 1, (board_size - 1) // 2),
            (board_size - 4, (board_size - 1) // 2)
        )
        for x, y in coords:
            ax.plot(
                x, y, 'o',
                markersize=config['center'][board_size],
                markeredgecolor='k', markerfacecolor='k',
                markeredgewidth=2
            )


def draw_plane(ax, plane, color, config: dict = {}, scale: float = 1.0):
    board_size = plane.shape[0]
    board_key = make_key(board_size)
    config = default_config | config
    markersize = config['marker'][board_key]
    ec, fc = ((.5, .5, .5), 'k') if int(color) == int(Color.BLACK) else ('k', 'w')
    for r, c in all_coordinates(board_size):
        if not plane[c][r]:
            continue
        ax.plot(r, c, 'o',
                markersize=markersize*scale, markeredgecolor=ec,
                markerfacecolor=fc, markeredgewidth=2*scale)


def place_stones(ax, state: State | cygo.State,
                 config: dict = {}, scale: float = 1.0):
    """place stones on ax

    :param ax: axis returned by `py:func:make_fig`
    :param state: :py:class:`migo.State` or :py:class:`cygo.State`

    >>> state = migo.State(9)
    >>> state.make_move((4, 4))
    False
    >>> fig, ax, _ = migo.drawing.make_fig(state.board_size)
    >>> migo.drawing.place_stones(ax, state)
    <Axes:...
    """
    planes = migo.features.board(state)
    black_index = (1 - int(state.current_player)) // 2
    draw_plane(ax, planes[black_index], Color.BLACK, config, scale)
    draw_plane(ax, planes[1 - black_index], Color.WHITE, config, scale)
    return ax


def put_number(ax, board_size, color, position, number,
               config: dict = {}, with_shade: bool = False,
               scale: float = 1.0):
    """put letters (typically digits) at position with color"""
    board_key = make_key(board_size)
    config = default_config | config
    r, c = position
    margin = config["margin"][board_key]
    fontsize = config["font"][board_key]
    others = {}
    if with_shade:
        others['bbox'] = dict(
            facecolor=config["facecolor"],
            alpha=0.8, edgecolor='none'
        )
    ax.text(r, c-margin, number, va='baseline', ha='center',
            fontsize=fontsize*scale, color=color, zorder=3,
            **others)


def plot_state(state, config: dict = {}):
    """make an image for given state
    """
    board_size = state.board_size
    fig, ax, _ = make_fig(board_size, config=config)
    planes = np.zeros((2, board_size, board_size))
    state_planes = migo.features.board(state)
    black_index = (1 - int(state.current_player)) // 2
    planes[0] = state_planes[black_index]
    planes[1] = state_planes[1 - black_index]
    draw_plane(ax, planes[0], Color.BLACK)
    draw_plane(ax, planes[1], Color.WHITE)
    return fig, ax


def plot_record(record, config: dict = {}, *,
                with_note: bool = True, overwrite: bool = True):
    """make an image for given record

    :param overwrite: the last stone takes precedence for each vertex if True
    """
    board_size = record.board_size
    state = cygo.State(board_size)
    fig, ax, ax_note = make_fig(board_size,
                                config=config, with_note=with_note)
    planes = np.zeros((2, board_size, board_size))
    if overwrite:
        cygo.apply_moves(state, record.moves)
        state_planes = migo.features.board(state)
        black_index = (1 - int(state.current_player)) // 2
        planes[0] = state_planes[black_index]
        planes[1] = state_planes[1 - black_index]
    else:
        for i, move in enumerate(record.moves):
            if move < 0:        # pass
                continue
            cmove = cygo.Move.from_raw_value(move, board_size=board_size)
            if planes[:, cmove.row, cmove.col].max() == 0:
                planes[i % 2, cmove.row, cmove.col] = 1
    draw_plane(ax, planes[0], Color.BLACK)
    draw_plane(ax, planes[1], Color.WHITE)

    coord_id_dic = {}
    for i, move in enumerate(record.moves):
        if move < 0:
            continue
        cmove = cygo.Move.from_raw_value(move, board_size=board_size)
        coord = (cmove.row, cmove.col)
        if coord not in coord_id_dic:
            coord_id_dic[coord] = []
        coord_id_dic[coord].append(i+1)
    omitted = []
    for coord, lst in coord_id_dic.items():
        color = 'k'             # empty or white
        if planes[0, coord[0], coord[1]]:
            color = 'w'
        id = lst[-1] if overwrite else lst[0]
        with_shade = planes[:, coord[0], coord[1]].max() == 0  # empty
        put_number(ax, board_size, color, coord, id,
                   config=config, with_shade=with_shade)
        omitted += [(_, id) for _ in lst if _ != id]

    if with_note:
        config = default_config | config
        board_key = make_key(board_size)
        fontsize = config["font"][board_key]
        for i in range(min(19, len(omitted))):
            a, b = omitted[i]
            label = f'{a} ({b})'
            ax_note.text(-0.5, 0.95-0.05*i, label, va='center', ha='left',
                         fontsize=fontsize, color='black')

    return fig, ax, ax_note
