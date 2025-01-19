from collections import defaultdict
from copy import deepcopy
from io import StringIO
from typing import Set, List, Dict, Union

import numpy as np

from migo.gtp import parse_move
from migo.hash import NeighborTable, ZobristHash
from migo.misc import all_coordinates, Color, Coord, Move, IllegalMoveError, \
    Pass, index_to_coord


class State:
    """state

    >>> state = migo.State(4)
    >>> print(state)
        A  B  C  D 
       ------------
     4| .  .  .  . |
     3| .  .  .  . |
     2| .  .  .  . |
     1| .  .  .  . |
       ------------
    ...
    >>> state.make_move((2, 1))
    False
    >>> state.make_move('c3')
    False
    >>> print(state)
        A  B  C  D 
       ------------
     4| .  .  .  . |
     3| .  B  W  . |
     2| .  .  .  . |
     1| .  .  .  . |
       ------------
    ...
    >>> state.liberty_count_of((2, 2))
    3
    >>> state.make_move('b3')
    False
    >>> state.liberty_count_of((2, 2))
    2
    """
    def __init__(self, board_size: int = 19, komi: float = 7.5,
                 superko_rule=True, max_history_n=7):
        self.neighbor_table = NeighborTable(size=board_size)

        self.__legal_moves_cache = None  # type: Set[Coord]
        self.__legal_eyes_cache = None  # type: Set[Coord]

        # 1 true, -1 false eye, 0 unknown
        self.__eyes_cache = np.zeros((board_size, board_size), dtype=int)

        self.board_size = board_size
        self.komi = komi
        self.superko_rule = superko_rule

        self.ko: Coord = None
        self.is_game_over = False
        self.current_player: Color = Color.BLACK
        self.passes: Dict[Color, int] = defaultdict(int)
        self.prisoners: Dict[Color, int] = defaultdict(int)

        self.board = np.zeros((board_size, board_size))
        self.board.fill(Color.EMPTY)

        self.board_hash = ZobristHash()
        self.hash_history = set()  # type: Set[int]
        self.move_history = []  # type: List[Move]

        self.groups: Dict[Coord, Set] = {
            coord: set() for coord in all_coordinates(board_size)
        }
        self.liberty_sets = {
            coord: set(self.crosswise_neighbors_of(coord))
            for coord in all_coordinates(board_size)
        }
        self.liberty_counts = np.zeros((board_size, board_size), dtype=int)
        self.liberty_counts.fill(-1)

        self.stone_ages = np.zeros((board_size, board_size), dtype=int)
        self.stone_ages.fill(-1)

        self.max_history_n = max_history_n
        self.history_buffer = []  # type: List[np.ndarray]

    def copy(self):
        other = State(board_size=self.board_size,
                      komi=self.komi,
                      superko_rule=self.superko_rule,
                      max_history_n=self.max_history_n)

        other.ko = self.ko
        other.is_game_over = self.is_game_over
        other.current_player = self.current_player
        other.passes = self.passes.copy()
        other.prisoners = self.prisoners.copy()

        other.board = self.board.copy()

        other.board_hash = ZobristHash(self.zobrist_hash)
        other.hash_history = self.hash_history.copy()
        other.move_history = self.move_history.copy()

        other.groups = deepcopy(self.groups)
        other.liberty_sets = deepcopy(self.liberty_sets)
        other.liberty_counts = self.liberty_counts.copy()
        other.stone_ages = self.stone_ages.copy()

        other.max_history_n = self.max_history_n
        other.history_buffer = self.history_buffer.copy()

        return other

    def legal_moves(self, include_eyes=True) -> Set[Coord]:
        if not self.__legal_moves_cache:
            self.__create_legal_move_cache()

        if include_eyes:
            return self.__legal_moves_cache | self.__legal_eyes_cache

        return set(self.__legal_moves_cache)

    def make_move(self, action: Union[Move, int, str], color: Color = None
                  ) -> bool:
        """place a stone of color at action

        :return: bool indicating game over
        """
        if isinstance(action, tuple) or action is None:
            pass
        elif isinstance(action, int):
            action = index_to_coord(action, self.board_size)
        elif isinstance(action, str):
            action = parse_move(action)
        else:
            raise IllegalMoveError("Expected tuple, int, or str but got `%s`"
                                   % type(action))

        if self.max_history_n <= 0:
            return self.__make_move_impl(action, color)

        board = self.board.copy()

        ret = self.__make_move_impl(action, color)

        self.history_buffer.append(board)

        while len(self.history_buffer) > self.max_history_n:
            self.history_buffer.pop(0)

        return ret

    def drop_history(self):
        self.history_buffer = []

    @property
    def zobrist_hash(self):
        """hash value of board in np.uint64"""
        return self.board_hash.value

    @property
    def empties(self) -> Set[Coord]:
        """set of coordinates without any stone"""
        return set(zip(*np.where(self.board == Color.EMPTY)))

    @property
    def group_list(self) -> List[Set[Coord]]:
        """list of groups covering all stones"""
        groups = []

        for coord in all_coordinates(self.board_size):
            group = self.groups[coord]

            if group and group not in groups:
                groups.append(group)

        return groups

    @property
    def liberty_set_list(self) -> List[Set[Coord]]:
        liberty_sets = []

        for coord in all_coordinates(self.board_size):
            liberty_set = self.liberty_sets[coord]

            if liberty_set and liberty_set not in liberty_sets:
                liberty_sets.append(liberty_set)

        return liberty_sets

    def color_at(self, position: Coord) -> Color:
        """return color at coord"""
        return Color(self.board[position])

    def group_at(self, position: Coord) -> Set[Coord]:
        """return group at position"""
        return set(self.groups[position])

    def groups_around_at(self, position: Coord) -> List[Set[Coord]]:
        """return groups around position"""
        groups = []

        for neighbor in self.crosswise_neighbors_of(position):
            group = self.groups[neighbor]
            if group and group not in groups:
                groups.append(group)

        return groups

    def stone_age_of(self, position: Coord) -> int:
        """return age of stone at position"""
        return self.stone_ages[position]

    def liberty_count_of(self, position: Coord) -> int:
        """count liberties at position"""
        return self.liberty_counts[position]

    def liberty_set_of(self, position: Coord) -> Set[Coord]:
        """set of liberties at position"""
        return set(self.liberty_sets[position])

    def crosswise_neighbors_of(self, position: Coord) -> List[Coord]:
        """four neighbors of position"""
        return self.neighbor_table.neighbors_crosswise(position)

    def diagonal_neighbors_of(self, position: Coord) -> List[Coord]:
        """four diagonal neighbors of position"""
        return self.neighbor_table.neighbors_diagonal(position)

    def is_on_board(self, position: Coord) -> bool:
        """whether position is valid"""
        x, y = position
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def is_eyeish(self, position: Coord, owner: Color) -> bool:
        """identify eye in a lightweight manner possibly with false positive"""
        if self.board[position] != Color.EMPTY:
            return False

        return all(self.board[neighbor] == owner
                   for neighbor in self.crosswise_neighbors_of(position))

    def is_eye(self, position: Coord, owner: Color,
               stack: list[Coord] | None = None
               ) -> bool:
        """identify whether position is eye for owner"""
        if not self.is_eyeish(position, owner):
            return False

        if stack is None:
            stack = list()
        elif position in stack:
            return True

        if cache := self.__eyes_cache[position[0], position[1]]:
            return True if cache > 0 else False

        other = owner.opponent()
        num_allowable_bad_coordinates = 1 \
            if len(self.crosswise_neighbors_of(position)) == 4 else 0

        num_bad_coordinates = 0
        for diagonal in self.diagonal_neighbors_of(position):
            if self.board[diagonal] == other:
                num_bad_coordinates += 1

            elif self.board[diagonal] == Color.EMPTY and diagonal not in stack:
                stack.append(position)
                if not self.is_eye(diagonal, owner, stack):
                    num_bad_coordinates += 1
                stack.pop()

            if num_bad_coordinates > num_allowable_bad_coordinates:
                self.__eyes_cache[position[0], position[1]] = -1
                return False

        self.__eyes_cache[position[0], position[1]] = 1
        return True

    def is_suicide_move(self, action: Move, color: Color | None = None
                        ) -> bool:
        """whether action is prohibited due to lack of liberties"""
        if action is Pass:
            return False

        position = action  # type: Coord
        color = color or self.current_player
        opponent_color = color.opponent()

        num_liberties = len(self.liberty_sets[action])

        if num_liberties != 0:
            return False

        for neighbor in self.crosswise_neighbors_of(position):
            group_has_other_liberties = \
                len(self.liberty_sets[neighbor] - {action}) > 0

            # if the move is saving another group
            if self.board[neighbor] == color and group_has_other_liberties:
                return False

            # if the move is killing an opponent group
            if self.board[neighbor] == opponent_color\
               and not group_has_other_liberties:
                return False

        return True

    def is_legal(self, action: Move, color: Color = None) -> bool:
        """identify action is legal"""
        if action is Pass:
            return True

        position = action  # type: Coord
        color = color or self.current_player

        if not self.is_on_board(position):
            return False
        if self.board[position] != Color.EMPTY:
            return False
        if position == self.ko:
            return False
        if self.is_suicide_move(position):  # intentionally ignore color
            return False
        if self.superko_rule and self.is_positional_superko(position, color):
            return False

        return True

    def is_positional_superko(self, position: Coord, color: Color = None
                              ) -> bool:
        """identify action is not allowed due to superko_rule"""
        color = color or self.current_player

        if color == Color.BLACK:
            player_history = self.move_history[0::2]
        elif color == Color.WHITE:
            player_history = self.move_history[1::2]
        else:
            raise ValueError("color should be either Color.BLACK"
                             " or Color.WHITE. got %s" % color)

        if position not in player_history:
            return False

        copied = self.copy()

        copied.superko_rule = False
        copied.make_move(position, color)

        return copied.zobrist_hash in self.hash_history

    def is_ladder_capture(self, position: Coord, prey: Coord = None,
                          remaining_attempts=80):
        """identify position is for ladder capture for prey"""
        if remaining_attempts <= 0:
            return False

        if not self.is_legal(position):
            return False

        hunter_color = self.current_player
        prey_color = self.current_player.opponent()

        if prey is None:
            neighbor_groups_stones = [
                next(iter(group)) for group in self.groups_around_at(position)
            ]
            potential_prey = [
                neighbor for neighbor in neighbor_groups_stones
                if (self.board[neighbor] == prey_color
                    and self.liberty_counts[neighbor] == 2)
            ]
            if not potential_prey:
                return False
        else:
            potential_prey = [prey]

        copied = self.copy()
        copied.make_move(position)

        for prey_coord in potential_prey:
            possible_escapes = set(copied.liberty_sets[prey_coord])

            for prey_stone in copied.groups[prey_coord]:
                for neighbor in copied.crosswise_neighbors_of(prey_stone):
                    if copied.board[neighbor] == hunter_color\
                       and copied.liberty_counts[neighbor] == 1:
                        # for each neighbor stone of the prey groups, check that the stone is a hunter stone and
                        # the hunter group is atari, then the group can be captured and
                        # possibly the prey can escape by capturing the group
                        possible_escapes |= copied.liberty_sets[neighbor]

            prey_can_escape = any(
                copied.is_ladder_escape(
                    escape_coord, prey=prey_coord,
                    remaining_attempts=(remaining_attempts - 1))
                for escape_coord in possible_escapes)

            if not prey_can_escape:
                return True

        return False

    def is_ladder_escape(self, position: Coord, prey: Coord = None,
                         remaining_attempts=80):
        """identify position is a candidate for ladder escape"""
        if remaining_attempts <= 0:
            return False

        if not self.is_legal(position):
            return False

        prey_color = self.current_player

        if prey is None:
            neighbor_groups_stones = [
                next(iter(group)) for group in self.groups_around_at(position)
            ]
            potential_prey = [
                neighbor for neighbor in neighbor_groups_stones
                if (self.board[neighbor] == prey_color
                    and self.liberty_counts[neighbor] == 1)
            ]

            if not potential_prey:
                return False
        else:
            potential_prey = [prey]

        copied = self.copy()
        copied.make_move(position)

        for prey_coord in potential_prey:
            if copied.liberty_counts[prey_coord] >= 3:
                return True

        for prey_coord in potential_prey:
            if copied.liberty_counts[prey_coord] == 1:
                continue

            possible_captures = copied.liberty_sets[prey_coord]

            hunter_can_capture = any(
                copied.is_ladder_capture(
                    possible_capture, prey=prey_coord,
                    remaining_attempts=(remaining_attempts - 1))
                for possible_capture in possible_captures)

            if hunter_can_capture:
                continue

            return True

        return False

    def __create_legal_move_cache(self):
        legal_moves = {
            move for move in all_coordinates(self.board_size)
            if self.is_legal(move)
        }

        self.__legal_moves_cache = {
            move for move in legal_moves
            if not self.is_eye(move, self.current_player)
        }
        self.__legal_eyes_cache = legal_moves - self.__legal_moves_cache

    def __clear_legal_move_cache(self):
        self.__legal_moves_cache = None
        self.__legal_eyes_cache = None
        self.__eyes_cache[:] = 0

    def __place_stone(self, position: Coord, color: Color):
        self.board[position] = color
        self.board_hash.update(position, color)

        self.stone_ages[position] = 0

        self.__update_neighbors(position, color)

    def __update_neighbors(self, position: Coord, owner: Color):
        other = owner.opponent()

        merged_group = {position}
        merged_liberty = self.liberty_sets[position]

        for neighbor in self.crosswise_neighbors_of(position):
            # `position` is no longer liberty
            self.liberty_sets[neighbor] -= {position}

            if self.board[neighbor] == other:
                # if the neighbor is enemy, reduce their liberty count
                new_liberty_count = len(self.liberty_sets[neighbor])

                for group in self.groups[neighbor]:
                    # propagate the liberty count
                    self.liberty_counts[group] = new_liberty_count

            elif self.board[neighbor] == owner:
                # merge friends
                merged_group |= self.groups[neighbor]
                merged_liberty |= self.liberty_sets[neighbor]

        merged_liberty_count = len(merged_liberty)
        # do merge
        for group in merged_group:
            self.groups[group] = merged_group
            self.liberty_sets[group] = merged_liberty
            self.liberty_counts[group] = merged_liberty_count

    def __remove_group(self, group: Set[Coord]):
        for position in group:
            self.board_hash.update(position, self.board[position])
            self.board[position] = Color.EMPTY

        for position in group:
            self.groups[position] = set()
            self.liberty_sets[position] = set()
            self.liberty_counts[position] = -1
            self.stone_ages[position] = -1

            for neighbor in self.crosswise_neighbors_of(position):
                if self.board[neighbor] == Color.EMPTY:
                    self.liberty_sets[position].add(neighbor)
                else:
                    self.liberty_sets[neighbor].add(position)

                    for group in self.groups[neighbor]:
                        self.liberty_counts[group] = len(self.liberty_sets[neighbor])

    def __make_move_impl(self, action: Move, color: Color = None) -> bool:
        color = color or self.current_player

        if not self.is_legal(action, color):
            raise IllegalMoveError("Cannot play %s by %s" % (action, color))

        self.move_history.append(action)
        self.ko = None
        self.stone_ages[self.stone_ages >= 0] += 1
        self.current_player = color.opponent()

        self.__clear_legal_move_cache()

        if action is Pass:
            self.passes[color] += 1

            if len(self.move_history) >= 2:
                self.is_game_over = (
                    self.move_history[-1] is Pass
                    and self.move_history[-2] is Pass
                )

            return self.is_game_over

        position = action  # type: Coord
        other = color.opponent()

        self.__place_stone(position, color)

        # check neighbors' liberty and remove if the move kills neighbors
        for neighbor in self.crosswise_neighbors_of(position):
            if self.board[neighbor] != other\
               or len(self.liberty_sets[neighbor]) != 0:
                # no capturing
                continue

            captured_group = self.groups[neighbor]
            num_captured_stones = len(captured_group)

            self.__remove_group(captured_group)

            self.prisoners[color] += num_captured_stones

            if num_captured_stones == 1:
                # check the move is ko or not
                would_recapture = len(self.liberty_sets[position]) == 1
                recapture_size = len(self.groups[position])

                if would_recapture and recapture_size == 1:
                    self.ko = neighbor

        # after calling __remove_group method, we can add the hash to the history
        # because __remove_group changes board_hash.
        self.hash_history.add(self.zobrist_hash)

        return self.is_game_over

    def __str__(self):
        buffer = StringIO()

        row_characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

        header = '   ' + ''.join(
            [" " + row_characters[i] + " " for i in range(self.board_size)]
        ) + "\n"

        buffer.write(header)

        buffer.write("   ")
        for x in range(self.board_size):
            buffer.write("---")
        buffer.write("\n")

        for y in range(self.board_size - 1, -1, -1):  # reverse(list(range(self.size))
            buffer.write("%2d|" % (y + 1))
            for x in range(self.board_size):
                color = self.color_at((y, x))

                if color == Color.BLACK:
                    character = 'B'
                elif color == Color.WHITE:
                    character = 'W'
                else:
                    character = '.'

                buffer.write(" %s " % character)
            buffer.write("|\n")

        buffer.write("   ")
        for x in range(self.board_size):
            buffer.write("---")
        buffer.write("\n")

        return buffer.getvalue()

    def to_cygo(self):
        """return corresponding :py:class:`cygo.State`

        .. warning::
           History is not preserved.

        .. warning::
           This is an experimental implementation,
           and may subject to change in future.

        >>> state, _ = migo.state.parse('. B . B|. B . B|. W W .|. . . .|')
        >>> print(state)
            A  B  C  D 
           ------------
         4| .  .  .  . |
         3| .  W  W  . |
         2| .  B  .  B |
         1| .  B  .  B |
           ------------
        ...
        >>> cstate = state.to_cygo()
        >>> print(cstate)
          A  B  C  D 
        4 .  .  .  . 4
        3 .  O (O) . 3
        2 .  X  .  X 2
        1 .  X  .  X 1
          A  B  C  D 
        ...
        >>> for v in migo.all_coordinates(state.board_size):
        ...     assert int(state.color_at(v)) == int(cstate.color_at(v))
        """
        import cygo
        cstate = cygo.State(
            self.board_size, self.komi, self.superko_rule, self.max_history_n
        )
        for vertex in all_coordinates(self.board_size):
            stone = self.color_at(vertex)
            if stone == Color.EMPTY:
                continue
            cstate.make_move(vertex, cygo.Color(stone))
        cstate.current_player = cygo.Color(self.current_player)
        return cstate


def parse(board_str: str, next_color: Color = None,
          keep_history: bool = False):
    """parse text board representation with optional move markers

    :param next_color: turn to move
    :param keep_history: (deprecated) keep pseudo make_moves as history if True
    :return: tuple of state and moves

    >>> board = ''.join([
    ...     'a B W . |',
    ...     'B B W . |',
    ...     '. B W W |',
    ...     '. B W . |',
    ... ])
    >>> state, moves = migo.state.parse(board)
    >>> print(state)
        A  B  C  D 
       ------------
     4| .  B  W  . |
     3| .  B  W  W |
     2| B  B  W  . |
     1| .  B  W  . |
       ------------
    ...
    >>> state.move_history
    []
    >>> len(moves)
    1
    >>> moves['a']
    (0, 0)
    >>> state.is_suicide_move(moves['a'], migo.Color.BLACK)
    False
    >>> state.is_suicide_move(moves['a'], migo.Color.WHITE)
    True
    """
    board_str = board_str.replace(' ', '')
    size = max(board_str.index('|'), board_str.count('|'))

    state = State(board_size=size)

    moves = {}
    place = state.make_move if keep_history else state._State__place_stone

    for row, row_str in enumerate(board_str.split('|')):
        for col, c in enumerate(row_str):
            if c == '.':
                continue

            if c in 'BX#':
                place((row, col), color=Color.BLACK)
            elif c in 'WO':
                place((row, col), color=Color.WHITE)
            else:
                moves[c] = (row, col)

    if next_color:
        state.current_player = next_color

    if not keep_history:
        state.hash_history.add(state.zobrist_hash)

    return state, moves
