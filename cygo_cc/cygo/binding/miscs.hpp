#ifndef CYGO_BINDING_MISCS_HPP
#define CYGO_BINDING_MISCS_HPP

#include "cygo/state.hpp"
#include "cygo/zobrist_hash.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace cygo {

void apply_moves(State& state, pybind11::array_t<int> const& moves, int move_id=-1);
template <class Vector>
void apply_moves_range(State& state, Vector const& moves, int first, int last);

ZobristHash::ValueType zobrist_hash(pybind11::array_t<float> const& black, pybind11::array_t<float> const& white);

}  // namespace cygo

template <class Vector>
void cygo::apply_moves_range(State& state, Vector const& moves, int first, int last) {
    auto board_size = state.board_size();
    for (int i = first; i < last; ++ i) {
        auto move = moves[i];

        if (move == -1) {
            state.make_move(Move::PASS);
        }
        else {
            state.make_move(Move::from_raw(move, board_size));
        }
    }
}


#endif //CYGO_BINDING_MISCS_HPP
