#include "cygo/binding/miscs.hpp"

#include "cygo/state.hpp"


namespace py = pybind11;

namespace cygo {

void apply_moves(State& state, py::array_t<int> const& moves) {
    auto board_size = state.board_size();
    auto move_length = moves.size();

    for (int i = 0; i < move_length; ++ i) {
        auto move = moves.at(i);

        if (move == -1) {
            state.make_move(Move::PASS);
        }
        else {
            state.make_move(Move::from_raw(move, board_size));
        }
    }
}

}  // namespace cygo