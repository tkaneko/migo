#ifndef CYGO_BINDING_MISCS_HPP
#define CYGO_BINDING_MISCS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cygo/zobrist_hash.hpp"


namespace cygo {

class State;

void apply_moves(State& state, pybind11::array_t<int> const& moves);

ZobristHash::ValueType zobrist_hash(pybind11::array_t<float> const& black, pybind11::array_t<float> const& white);

}  // namespace cygo

#endif //CYGO_BINDING_MISCS_HPP
