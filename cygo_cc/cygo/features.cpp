#include "features.hpp"

void cygo::feature_impl::batch_features_to_ptr(std::vector<State> const& state_list, int history_n,
                           int channel_size, int size_per_state,
                           int8_t *data, int8_t *legals_out) {
    auto task = [&](int first, int last) {
        for (int i = first; i < last; ++i) {
            const auto& state = state_list[i];
            feature_impl::store_history_n(state,
                                          static_cast<std::size_t>(history_n),
                                          &data[i*size_per_state]);
            // the last plane of state i
            auto side_plane_offset = (i + 1) * size_per_state - channel_size;
            fill_side_to_move(state, &data[side_plane_offset], channel_size);

            const auto& stones_b = state.black_board();
            const auto& stones_w = state.white_board();
            const int base = i * (channel_size + 1);
            for (size_t j = 0; j < stones_b.size(); ++j) {
                int id = base + j;
                legals_out[id] = 1 - (stones_b[j] + stones_w[j]);
            }
            legals_out[base + channel_size] = 1;
        }
    };
    task(0, state_list.size());
}

