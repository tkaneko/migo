#include "cygo/binding/features.hpp"
#include "cygo/binding/miscs.hpp"

#include "cygo/state.hpp"
#include "cygo/features.hpp"

#include <stdexcept>
#include <thread>

namespace py = pybind11;

namespace cygo {

template <typename T, typename TInt>
py::array_t<T> make_pyarray(T* array_ptr, std::initializer_list<TInt> shape) {
    return py::array_t<T>(shape, array_ptr, py::capsule(array_ptr, [](void* f) {
        delete[] reinterpret_cast<T*>(f);
    }));
}

template <typename T>
T* to_ptr(std::vector<T> const& v) {
    auto ptr = new T[v.size()]();

    std::copy(std::begin(v), std::end(v), ptr);

    return ptr;
}

}  // namespace cygo

py::array_t<float> cygo::black(State const& state) {
    return make_pyarray(to_ptr(feature_impl::color<float>(state, Color::BLACK)),
                        {1, state.board_size(), state.board_size()});
}

py::array_t<float> cygo::white(State const& state) {
    return make_pyarray(to_ptr(feature_impl::color<float>(state, Color::WHITE)),
                        {1, state.board_size(), state.board_size()});
}


py::array_t<float> cygo::board_i(State const& state, int i, Color c) {
    auto const length = state.max_history_n();

    if (i < 0 or length < static_cast<std::size_t>(i)) {
        std::ostringstream os;
        os << "Given index (" << i << ") is out of bound. Expected in [0, " << length << "]";

        throw std::invalid_argument(os.str());
    }

    if (c == Color::EMPTY) {
        return make_pyarray(to_ptr(feature_impl::board_i<float>(state, static_cast<std::size_t>(i))),
                            {2, state.board_size(), state.board_size()});
    }
    else {
        return make_pyarray(to_ptr(feature_impl::board_i_color<float>(state, static_cast<std::size_t>(i), c)),
                            {1, state.board_size(), state.board_size()});
    }
}

py::array_t<float> cygo::history_n(State const& state, int n, Color c) {
    auto const length = state.max_history_n();

    if (n < 0 or length < static_cast<std::size_t>(n)) {
        std::ostringstream os;
        os << "Given index (" << n << ") is out of bound. Expected in [0, " << length << "]";

        throw std::invalid_argument(os.str());
    }

    if (c == Color::EMPTY) {
        return make_pyarray(to_ptr(feature_impl::history_n<float>(state, static_cast<std::size_t>(n))),
                            {2 * (n + 1), state.board_size(), state.board_size()});
    }
    else {
        return make_pyarray(to_ptr(feature_impl::history_n_color<float>(state, static_cast<std::size_t>(n), c)),
                            {n + 1, state.board_size(), state.board_size()});
    }
}

namespace cygo
{
    template <typename Ptr>
    void fill_side_to_move(const State& state, Ptr ptr, int size) {
        auto value = (state.current_player == Color::BLACK) ? 1 : 0;
        std::fill_n(ptr, size, value);
    }
}  // namespace cygo


pybind11::array_t<float>
cygo::features_at(int board_size,
                  py::array_t<int16_t> const& moves, py::array_t<int> const& ids_arr,
                  int history_n) {
    auto ids = ids_arr.unchecked<1>();
    auto n_channels = 2 * (history_n + 1) + 1; // 1 for color (side to move)
    auto channel_size = board_size * board_size;
    auto size_per_state = n_channels * channel_size;
    py::array_t<float> ret(ids.size() * size_per_state);
    auto data = ret.mutable_unchecked<1>();
    
    auto id_length = ids.size();
    for (int i = 0; i < id_length; ++i) {
        State state(board_size, 7.5, true, history_n);
        apply_moves(state, moves, ids[i]);
        feature_impl::store_history_n(state, static_cast<std::size_t>(history_n),
                                      &data[i*size_per_state]);
        // the last plane of state i
        auto side_plane_offset = (i + 1) * size_per_state - channel_size;
        fill_side_to_move(state, &data[side_plane_offset], channel_size);
    }
    return ret.reshape({(int)ids.size() * n_channels, board_size, board_size});
}

namespace cygo
{
    const auto min_task_size = 128;
    const auto max_threads = 8;

    template <class F>
    void run_in_parallel(F task, int N) {
        std::vector<std::thread> workers;
        const auto n_parallel = std::min(max_threads, std::max(1, N / min_task_size));
        int task_size = N / n_parallel + 15;
        task_size = ((task_size + 15) / 16) * 16;
        for (int i=0; i<n_parallel; ++i) {
            auto first = std::min(task_size*i, N), last = std::min(task_size*(i+1), N);
            workers.emplace_back(task, first, last);
        }
        for (auto& w: workers)
            w.join();
    }
}


std::tuple<py::array_t<int8_t>, py::array_t<int16_t>, py::array_t<int8_t>>
cygo::collate(pybind11::array_t<int> const& indices_py,
              int history_n, int board_size,
              pybind11::array_t<int32_t> const& move_offset_py,
              pybind11::array_t<int16_t> const& game_moves_py,
              pybind11::array_t<int8_t> const& winner_py,
              pybind11::array_t<int32_t> const& data_offset_py,
              int ignore_opening_moves,
              bool correct_invalid_index
              ) {
    auto indices = indices_py.unchecked<1>();
    auto move_offset = move_offset_py.unchecked<1>();
    auto game_moves = game_moves_py.unchecked<1>();
    auto winner = winner_py.unchecked<1>();
    auto data_offset = data_offset_py.unchecked<1>();

    int N = indices.size();
    auto n_channels = 2 * (history_n + 1) + 1;
    auto channel_size = board_size * board_size;
    auto size_per_state = n_channels * channel_size;

    py::array_t<int8_t> ret_x_py(N * size_per_state);
    auto ret_x = ret_x_py.mutable_unchecked<1>();
    py::array_t<int16_t> move_labels_py(N);
    py::array_t<int8_t> value_labels_py(N);
    auto move_labels = move_labels_py.mutable_unchecked<1>();
    auto value_labels = value_labels_py.mutable_unchecked<1>();
    const auto data_limit = data_offset[data_offset.size()-1];

    auto task = [&](int first, int last){
        for (int i=first; i<last; ++i) {
            auto flat_idx = indices[i];
            if (flat_idx >= data_limit) {
                if (! correct_invalid_index)
                    throw std::domain_error("collate: flat_index exceeds limit "
                                            + std::to_string(flat_idx)
                                            +" >= "+std::to_string(data_limit));
                flat_idx %= data_limit;
            }
            auto [gid, move_id]
              = feature_impl::to_game_move_pair(data_offset, flat_idx,
                                                ignore_opening_moves);
            auto offset = move_offset[gid];
        
            State state(board_size, 7.5, false, history_n);
            apply_moves_range(state, game_moves, offset, offset+move_id);

            feature_impl::store_history_n(state, static_cast<std::size_t>(history_n),
                                          &ret_x[i * size_per_state]);
            // the last plane of state i
            auto side_plane_offset = (i + 1) * size_per_state - channel_size;
            fill_side_to_move(state, &ret_x[side_plane_offset], channel_size);

            move_labels[i] = game_moves[offset+move_id];
            auto sgn = (move_id % 2 == 0) ? 1 : -1;
            value_labels[i] = winner[gid] * sgn;
        }
    };
#if 1
    run_in_parallel(task, N);
#else
    task(0, N);
#endif
    return {
      ret_x_py.reshape({N, n_channels, board_size, board_size}),
      move_labels_py,
      value_labels_py
    };
}

std::tuple<pybind11::array_t<int8_t>, pybind11::array_t<int16_t>, pybind11::array_t<int8_t>, pybind11::array_t<float_t>>
cygo::collatez(pybind11::array_t<int> const& indices_py,
               int history_n, int board_size,
               pybind11::array_t<int32_t> const& move_offset_py,
               pybind11::array_t<int16_t> const& game_moves_py,
               pybind11::array_t<int8_t> const& winner_py,
               pybind11::array_t<int32_t> const& data_offset_py,
               int ignore_opening_moves,
               // for zone
               pybind11::array_t<int8_t> const& zones_py, // board * board * 2
               pybind11::array_t<float_t> const& zone_score_py, // games
               //
               bool correct_invalid_index
               )
{
    auto indices = indices_py.unchecked<1>();
    auto move_offset = move_offset_py.unchecked<1>();
    auto game_moves = game_moves_py.unchecked<1>();
    auto winner = winner_py.unchecked<1>();
    auto data_offset = data_offset_py.unchecked<1>();
    auto zones = zones_py.unchecked<1>();
    auto zone_score = zone_score_py.unchecked<1>();
    const auto data_limit = data_offset[data_offset.size()-1];

    auto channel_size = board_size * board_size;
    int N = indices.size();
    if (winner.size() != zone_score.size()
        || 2*winner.size()*channel_size != zones.size()) {
        auto msg = "inconsistent size in collatez "
            + std::to_string(N)
            + " " + std::to_string(zones.size())
            + " " + std::to_string(zone_score.size());
        throw std::invalid_argument(msg);
    }
    auto n_channels = 2 * (history_n + 1) + /* color */ 1 + /* zone */ 1;
    auto size_per_state = n_channels * channel_size;

    py::array_t<int8_t> ret_x_py(N * size_per_state);
    auto ret_x = ret_x_py.mutable_unchecked<1>();
    py::array_t<int16_t> move_labels_py(N);
    py::array_t<int8_t> value_labels_py(N);
    py::array_t<float_t> zone_labels_py(N);
    auto move_labels = move_labels_py.mutable_unchecked<1>();
    auto value_labels = value_labels_py.mutable_unchecked<1>();
    auto zone_labels = zone_labels_py.mutable_unchecked<1>();

    auto task = [&](int first, int last){
        for (int i=first; i<last; ++i) {
            auto flat_idx = indices[i];
            if (flat_idx >= data_limit) {
                if (! correct_invalid_index)
                    throw std::domain_error("collatez: flat_index exceeds limit "
                                            +std::to_string(flat_idx)
                                            +" >= "+std::to_string(data_limit)
                                            );
                flat_idx %= data_limit;
            }
            auto [gid, move_id]
              = feature_impl::to_game_move_pair(data_offset, flat_idx,
                                                ignore_opening_moves);
            auto offset = move_offset[gid];
        
            State state(board_size, 7.5, false, history_n);
            apply_moves_range(state, game_moves, offset, offset+move_id);

            feature_impl::store_history_n(state, static_cast<std::size_t>(history_n),
                                          &ret_x[i * size_per_state]);
            // the second last plane of state i
            auto side_plane_offset = (i + 1) * size_per_state - 2 * channel_size;
            fill_side_to_move(state, &ret_x[side_plane_offset], channel_size);
            // the last plane of state i
            auto zone_plane_offset = side_plane_offset + channel_size;
            auto zone_idx = gid * 2 + (move_id % 2);
            
            std::copy(&zones[zone_idx*channel_size], &zones[(zone_idx+1)*channel_size],
                      &ret_x[zone_plane_offset]);

            move_labels[i] = game_moves[offset+move_id];
            auto sgn = (move_id % 2 == 0) ? 1 : -1;
            value_labels[i] = winner[gid] * sgn;
            zone_labels[i] = zone_score[gid] * sgn;
        }
    };

#if 1
    run_in_parallel(task, N);
#else
    task(0, N);
#endif

    return {
        ret_x_py.reshape({N, n_channels, board_size, board_size}),
        move_labels_py,
        value_labels_py,
        zone_labels_py
    };
}

std::tuple<pybind11::array_t<int8_t>, pybind11::array_t<int16_t>, pybind11::array_t<int8_t>, pybind11::array_t<int8_t>, pybind11::array_t<float_t>>
cygo::collate_ext(pybind11::array_t<int> const& indices_py,
                  int history_n, int board_size,
                  pybind11::array_t<int32_t> const& move_offset_py,
                  pybind11::array_t<int16_t> const& game_moves_py,
                  pybind11::array_t<int8_t> const& winner_py, // games
                  pybind11::array_t<int32_t> const& data_offset_py,
                  int ignore_opening_moves,
                  // for zone
                  std::vector<int8_t> const& enabled_colors,
                  pybind11::array_t<int8_t> const& zones_py, // 2 * (board ** 2)
                  pybind11::array_t<int8_t> const& aux_plane_labels_py, // games * (board ** 2 + 1)
                  pybind11::array_t<float_t> const& aux_values_py // games
               )
{
    auto indices = indices_py.unchecked<1>();
    auto move_offset = move_offset_py.unchecked<1>();
    auto game_moves = game_moves_py.unchecked<1>();
    auto winner = winner_py.unchecked<1>();
    auto data_offset = data_offset_py.unchecked<1>();
    auto zones = zones_py.unchecked<1>();
    auto aux_plane_labels = aux_plane_labels_py.unchecked<1>();
    auto aux_values = aux_values_py.unchecked<1>();
    const auto data_limit = data_offset[data_offset.size()-1];

    auto channel_size = board_size * board_size;
    const auto oplane_dim = channel_size + 1; // +1 for pass
    int N = indices.size();
    if (2*winner.size() != aux_values.size()
        || 2*channel_size != zones.size()
        || winner.size()*oplane_dim != aux_plane_labels.size()) {
        auto msg = "inconsistent size in collate_ext "
            " #ch " + std::to_string(channel_size)
            + " #n " + std::to_string(N)
            + " #w " + std::to_string(winner.size())
            + " #zp " + std::to_string(zones.size())
            + " #zs " + std::to_string(aux_values.size());
        throw std::invalid_argument(msg);
    }
    if (enabled_colors.size() != 2)
        throw std::invalid_argument("collate_ext enabled_colors");
    auto n_channels = 2 * (history_n + 1) + /* color */ 1 + /* zone */ 1;
    auto size_per_state = n_channels * channel_size;

    py::array_t<int8_t> ret_x_py(N * size_per_state);
    auto ret_x = ret_x_py.mutable_unchecked<1>();
    py::array_t<int16_t> move_labels_py(N);
    py::array_t<int8_t> value_labels_py(N);
    py::array_t<float_t> aux_labels_py(N);
    py::array_t<int8_t> aux_oplanes_py(N * oplane_dim);
    auto move_labels = move_labels_py.mutable_unchecked<1>();
    auto value_labels = value_labels_py.mutable_unchecked<1>();
    auto aux_labels = aux_labels_py.mutable_unchecked<1>();
    auto aux_oplanes = aux_oplanes_py.mutable_unchecked<1>();
    std::fill(&aux_oplanes[0], &aux_oplanes[0]+aux_oplanes.size(), 0);

    auto task = [&](int first, int last){
        for (int i=first; i<last; ++i) {
            auto flat_idx = indices[i];
            if (flat_idx >= data_limit) {
              throw std::domain_error("collate_ext: flat_index exceeds limit "
                                      +std::to_string(flat_idx)
                                      +" >= "+std::to_string(data_limit)
                                      );
            }
            auto [gid, move_id]
              = feature_impl::to_game_move_pair(data_offset, flat_idx,
                                                ignore_opening_moves);
            if (! enabled_colors[move_id % 2])
                move_id = (move_id > 0) ? (move_id - 1) : (move_id + 1);
            auto offset = move_offset[gid];
        
            State state(board_size, 7.5, false, history_n);
            apply_moves_range(state, game_moves, offset, offset+move_id);
            feature_impl::store_history_n(state, static_cast<std::size_t>(history_n),
                                          &ret_x[i * size_per_state]);
            // the second last plane of state i
            auto side_plane_offset = (i + 1) * size_per_state - 2 * channel_size;
            fill_side_to_move(state, &ret_x[side_plane_offset], channel_size);
            // the last plane of state i
            auto zone_plane_offset = side_plane_offset + channel_size;
            auto zone_idx = move_id % 2;
            
            std::copy(&zones[zone_idx*channel_size], &zones[(zone_idx+1)*channel_size],
                      &ret_x[zone_plane_offset]);

            move_labels[i] = game_moves[offset+move_id];
            auto sgn = (move_id % 2 == 0) ? 1 : -1;
            value_labels[i] = winner[gid] * sgn;
            aux_labels[i] = aux_values[zone_idx*winner.size() + gid];

            // aux_oplanes
            // need to convert (black, white) into (me, opponent)
            auto dst = i*oplane_dim, src = gid*oplane_dim;
            if (move_id % 2)
                for (int v=0; v<channel_size; ++v)
                    aux_oplanes[dst + v] = -aux_plane_labels[src + v];
            else
                for (int v=0; v<channel_size; ++v)
                    aux_oplanes[dst + v] = aux_plane_labels[src + v];
        }
    };

#if 1
    run_in_parallel(task, N);
#else
    task(0, N);
#endif
    return {
        ret_x_py.reshape({N, n_channels, board_size, board_size}),
        move_labels_py,
        value_labels_py,
        aux_oplanes_py.reshape({N, oplane_dim}),
        aux_labels_py
    };
}

pybind11::array_t<int8_t>
cygo::make_territory(int board_size,
                     pybind11::array_t<int32_t> const& move_offset_py,
                     pybind11::array_t<int16_t> const& game_moves_py
                     )
{
    const auto channel_size = board_size * board_size;
    const auto oplane_dim = channel_size + 1; // +1 for pass
    auto move_offset = move_offset_py.unchecked<1>();
    auto game_moves = game_moves_py.unchecked<1>();
    const auto N = move_offset.size() - 1;
    py::array_t<int8_t> aux_oplanes_py(N * oplane_dim);
    auto aux_oplanes = aux_oplanes_py.mutable_unchecked<1>();
    std::fill(&aux_oplanes[0], &aux_oplanes[0]+aux_oplanes.size(), 0);
    auto task = [&](int first, int last) {
        for (int gid=first; gid < last; ++gid) {
            State state(board_size, 7.5, false, 0);
            auto first = move_offset[gid], last = move_offset[gid+1];
            apply_moves_range(state, game_moves, first, last);
            state.tromp_taylor_score(Color::EMPTY, &aux_oplanes[gid*oplane_dim]);
        }
    };
#if 1
    run_in_parallel(task, N);
#else
    task(0, N);
#endif
    return aux_oplanes_py;
}


std::pair<pybind11::array_t<int8_t>, pybind11::array_t<int8_t>>
cygo::batch_features(std::vector<State> const& state_list, int history_n) {
    int batch_size = state_list.size();
    if (batch_size == 0)
        return {};
    auto board_size = state_list[0].board_size();
    auto n_channels = 2 * (history_n + 1) + 1; // 1 for color (side to move)
    auto channel_size = board_size * board_size;
    auto size_per_state = n_channels * channel_size;
    py::array_t<float> ret(batch_size * size_per_state);
    py::array_t<int8_t> legals_relaxed_py(batch_size * (channel_size + 1));
    auto data = ret.mutable_unchecked<1>();
    auto legals_relaxed = legals_relaxed_py.mutable_unchecked<1>();

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
                assert (0 <= id && id < legals_relaxed_py.size());
                legals_relaxed[id] = 1 - (stones_b[j] + stones_w[j]);
            }
            legals_relaxed[base + channel_size] = 1;
        }
    };
    task(0, batch_size);
    return {
        ret.reshape({batch_size, n_channels, board_size, board_size}),
        legals_relaxed_py.reshape({batch_size, channel_size+1})
    };
}

std::pair<pybind11::array_t<int8_t>, pybind11::array_t<int8_t>>
cygo::batch_features_with_zone(std::vector<State> const& state_list, int history_n,
                               std::vector<pybind11::array_t<int8_t>> & zone_py) {
    int batch_size = state_list.size();
    if (batch_size == 0)
        return {};
    auto board_size = state_list[0].board_size();
    auto n_channels = 2 * (history_n + 1) + 1 + 1; // 1 for color (side to move), 1 for zone
    auto channel_size = board_size * board_size;
    auto size_per_state = n_channels * channel_size;
    py::array_t<float> ret(batch_size * size_per_state);
    py::array_t<int8_t> legals_relaxed_py(batch_size * (channel_size + 1));
    auto data = ret.mutable_unchecked<1>();
    auto legals_relaxed = legals_relaxed_py.mutable_unchecked<1>();

    auto task = [&](int first, int last) {
        for (int i = first; i < last; ++i) {
            const auto& state = state_list[i];
            feature_impl::store_history_n(state,
                                          static_cast<std::size_t>(history_n),
                                          &data[i*size_per_state]);
            // the last plane of state i
            auto side_plane_offset = (i + 1) * size_per_state - channel_size*2;
            fill_side_to_move(state, &data[side_plane_offset], channel_size);

            auto zone_plane_offset = side_plane_offset + channel_size;            
            auto zone = zone_py[i].reshape({channel_size}).unchecked<uint8_t,1>();
            std::copy(&zone[0], &zone[channel_size], &data[zone_plane_offset]);

            const auto& stones_b = state.black_board();
            const auto& stones_w = state.white_board();
            const int base = i * (channel_size + 1);
            for (size_t j = 0; j < stones_b.size(); ++j) {
                int id = base + j;
                assert (0 <= id && id < legals_relaxed_py.size());
                legals_relaxed[id] = 1 - (stones_b[j] + stones_w[j]);
            }

            legals_relaxed[base + channel_size] = 1;
        }
    };
    task(0, batch_size);
    return {
        ret.reshape({batch_size, n_channels, board_size, board_size}),
        legals_relaxed_py.reshape({batch_size, channel_size+1})
    };
}

