#include "cygo/binding/miscs.hpp"

#include "cygo/state.hpp"


namespace py = pybind11;

namespace cygo {

void apply_moves(State& state, py::array_t<int> const& moves_py, int move_id) {
    auto moves = moves_py.unchecked<1>();
    if (move_id >= moves.size())
        throw std::domain_error("apply_moves: range error in move_id "+std::to_string(move_id));

    int move_length = (move_id < 0) ? (int)moves.size() : move_id;
    apply_moves_range(state, moves, 0, move_length);
}

ZobristHash::ValueType zobrist_hash(pybind11::array_t<float> const& black, pybind11::array_t<float> const& white) {
    if (black.ndim() != 2 or white.ndim() != 2) {
        throw std::invalid_argument("ndim must be 2");
    }

    if (black.shape(0) != white.shape(0) or black.shape(1) != white.shape(1) or black.shape(0) != black.shape(1)) {
        throw std::invalid_argument("all dimensions should have a same length");
    }

    auto board_size = static_cast<int>(black.shape(0));

    std::vector<Move> black_indices, white_indices;

    for (int i = 0; i < board_size; ++i) {
        for (int j = 0; j < board_size; ++j) {
            if (black.at(i, j) == 1.0) {
                black_indices.push_back(Move::from_coordinate(i, j, board_size));
            }
            else if (white.at(i, j) == 1.0) {
                white_indices.push_back(Move::from_coordinate(i, j, board_size));
            }
        }
    }

    return ZobristHash::calculate_hash(black_indices, white_indices);
}

}  // namespace cygo
