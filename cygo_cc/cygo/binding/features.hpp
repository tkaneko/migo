#ifndef CYGO_BINDING_FEATURES_HPP
#define CYGO_BINDING_FEATURES_HPP

#include <list>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cygo/color.hpp"


namespace cygo {

class State;

pybind11::array_t<float> board_i(State const& state, int i, Color c);

pybind11::array_t<float> history_n(State const& state, int n, Color c);

pybind11::array_t<float> black(State const& state);

pybind11::array_t<float> white(State const& state);

pybind11::array_t<float>
features_at(int board_size, pybind11::array_t<int16_t> const& moves, pybind11::array_t<int> const& ids,
            int history_n);


std::tuple<pybind11::array_t<int8_t>, pybind11::array_t<int16_t>, pybind11::array_t<int8_t>>
collate(pybind11::array_t<int> const& indices,
        int history_n, int board_size,
        pybind11::array_t<int32_t> const& move_offset,
        pybind11::array_t<int16_t> const& game_moves,
        pybind11::array_t<int8_t> const& winner,
        pybind11::array_t<int32_t> const& data_offset,
        int ignore_opening_moves,
        bool correct_invalid_index=false
        );

std::tuple<pybind11::array_t<int8_t>, pybind11::array_t<int16_t>, pybind11::array_t<int8_t>, pybind11::array_t<float_t>>
collatez(pybind11::array_t<int> const& indices,
         int history_n, int board_size,
         pybind11::array_t<int32_t> const& move_offset,
         pybind11::array_t<int16_t> const& game_moves,
         pybind11::array_t<int8_t> const& winner,
         pybind11::array_t<int32_t> const& data_offset,
         int ignore_opening_moves,
         // for zone
         pybind11::array_t<int8_t> const& zones, // board * board * 2
         pybind11::array_t<float_t> const& zone_score, // games
         // args with default value
         bool correct_invalid_index=false
         );

std::tuple<pybind11::array_t<int8_t>, pybind11::array_t<int16_t>, pybind11::array_t<int8_t>, pybind11::array_t<int8_t>, pybind11::array_t<float_t>>
collate_ext(pybind11::array_t<int> const& indices,
            int history_n, int board_size,
            pybind11::array_t<int32_t> const& move_offset,
            pybind11::array_t<int16_t> const& game_moves,
            pybind11::array_t<int8_t> const& winner,
            pybind11::array_t<int32_t> const& data_offset,
            int ignore_opening_moves,
            // for aux
            std::vector<int8_t> const& enabled_colors,
            pybind11::array_t<int8_t> const& zones, // board * board * 2
            pybind11::array_t<int8_t> const& aux_plane_labels, // games
            pybind11::array_t<float_t> const& zone_score // games
            );

pybind11::array_t<int8_t>
make_territory(int board_size,
               pybind11::array_t<int32_t> const& move_offset,
               pybind11::array_t<int16_t> const& game_moves
               );

std::pair<pybind11::array_t<int8_t>, pybind11::array_t<int8_t>>
batch_features(std::vector<State> const& state_list, int history_n);

std::pair<pybind11::array_t<int8_t>, pybind11::array_t<int8_t>>
batch_features_with_zone(std::vector<State> const& state_list, int history_n,
                         std::vector<pybind11::array_t<int8_t>> & zone_list);

  
}  // namespace cygo

#endif //CYGO_BINDING_FEATURES_HPP
