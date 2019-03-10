#ifndef CYGO_BINDING_MISCS_HPP
#define CYGO_BINDING_MISCS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace cygo {

class State;

void apply_moves(State& state, pybind11::array_t<int> const& moves);

}  // namespace cygo

#endif //CYGO_BINDING_MISCS_HPP
