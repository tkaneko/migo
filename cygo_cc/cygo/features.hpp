#ifndef CYGO_IMPL_FEATURES_HPP
#define CYGO_IMPL_FEATURES_HPP

#include <algorithm>
#include <thread>
#include <cassert>

#include "color.hpp"
#include "state.hpp"


namespace cygo {
namespace feature_impl {
/** unsafe implementation with raw pointers, see batch_features in binding/features.cpp */
void batch_features_to_ptr(std::vector<State> const& state_list, int history_n,
                           int channel_size, int size_per_state, int8_t* data,
                           int8_t* legals_out);

/** unsafe implementation with raw pointers, see batch_features_with_zone binding/features.cpp */
template <typename Zone>
void batch_features_with_zone_to_ptr(
    std::vector<State> const& state_list, int history_n,
    Zone zone_seq,  // idx -> pointer to const int8[channel_size]
    int channel_size, int size_per_state, int8_t* data, int8_t* legals_out);

template <typename Ptr>
void fill_side_to_move(const State& state, Ptr ptr, int size) {
    auto value = (state.current_player == Color::BLACK) ? 1 : 0;
    std::fill_n(ptr, size, value);
}

template <typename T>
std::vector<T> color(State const& state, Color c) {
    assert(c != Color::EMPTY);

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto array_size = board_size * board_size;

    std::vector<T> ret(array_size, T(0));

    if (state.current_player == c) {
        std::fill_n(std::begin(ret), array_size, T(1));
    }

    return ret;
}

template <typename T>
std::vector<T> board_i_color(State const& state, std::size_t i, Color c) {
    // Returns the color's plane of S at T_{t-i}.
    // The shape of the resultant array is (1, size, size)
    assert(c != Color::EMPTY);

    auto const& history = state.history(c);

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size;

    std::vector<T> ret(array_size, T(0));

    if (history.size() <= i) {
        return ret;
    }

    auto itr = std::begin(history);

    std::advance(itr, i);
    std::copy(std::begin(*itr), std::end(*itr), std::begin(ret));

    return ret;
}

template <typename T>
std::vector<T> board_i(State const& state, std::size_t i) {
    assert(0 <= (int)i and i <= state.max_history_n());

    // Returns the plane of S at T_{t-i}
    // shape: (2, size, size)
    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * 2;

    std::vector<T> ret(array_size, T(0));

    auto const& history_1 = state.history(state.current_player);
    auto const& history_2 = state.history(opposite_color(state.current_player));

    assert(history_1.size() == history_2.size());

    if (history_1.size() <= i or history_2.size() <= i) {
        return ret;
    }

    auto itr_1 = std::begin(history_1);
    auto itr_2 = std::begin(history_2);

    std::advance(itr_1, i);
    std::advance(itr_2, i);

    auto const& board_1 = *itr_1;
    auto const& board_2 = *itr_2;

    std::copy(std::begin(board_1), std::end(board_1), std::begin(ret));
    std::copy(std::begin(board_2), std::end(board_2), std::next(std::begin(ret), plane_size));

    return ret;
}

template <typename T>
std::vector<T> history_n_color(State const& state, std::size_t n, Color c) {
    assert(0 <= (int)n and n <= state.max_history_n());

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * (n + 1);

    std::vector<T> ret(array_size, T(0));

    auto const& history = state.history(c);
    auto itr = std::begin(history);

    std::size_t i = 0;
    auto length = history.size();

    auto ret_itr = std::begin(ret);

    while (i <= n and i < length) {
        auto const &board = *itr;
        std::copy(std::begin(board), std::end(board), ret_itr);

        ++itr;
        std::advance(ret_itr, plane_size);
        i += 1;
    }

    return ret;
}

template <typename Iter>
void store_history_n(State const& state, std::size_t n, Iter out) {
    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;

    auto const& history_1 = state.history(state.current_player);
    auto const& history_2 = state.history(opposite_color(state.current_player));

    assert(history_1.size() == history_2.size());

    auto itr_1 = std::begin(history_1);
    auto itr_2 = std::begin(history_2);

    std::size_t i = 0;
    auto length = history_1.size();

    while (i <= n and i < length) {
        auto const& board_1 = *itr_1;
        auto const& board_2 = *itr_2;

        std::copy(std::begin(board_1), std::end(board_1), out);
        std::copy(std::begin(board_2), std::end(board_2), std::next(out, plane_size));

        ++itr_1; ++itr_2;
        std::advance(out, 2 * plane_size);
        i += 1;
    }
    if (i <= n) {
        // when history is shorter than length, typically near opening games
        std::fill(out, out + (n + 1 - i) * 2 * plane_size, 0);
    }
}
  

template <typename T>
std::vector<T> history_n(State const& state, std::size_t n) {
    assert(0 <= (int)n and n < state.max_history_n());

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * 2 * (n + 1);

    std::vector<T> ret(array_size, T(0));

    auto ret_itr = std::begin(ret);
    store_history_n(state, n, ret_itr);

    return ret;
}

    template <class Vector>
    std::pair<int, int> to_game_move_pair(const Vector& game_index, int flat_index,
                                          int ignore_opening_moves) {
        auto first = &game_index[0], last = &game_index[0]+game_index.size();
        auto p = std::upper_bound(first, last, flat_index);
        --p;
        int gid = p - first;
        int mid = flat_index - *p;
        if (gid < 0 || mid < 0)
            throw std::logic_error("to_game_move_pair: panic");
        return {gid, mid + ignore_opening_moves};
    }

    template<typename T>
    std::vector<T> id_plane(int board_size, int id, int model_n) {
        if (id < 0 || id >= model_n) {
            throw std::out_of_range("id is out of range");
        }

        // shape = (model_n, board_size, board_size)
        std::vector<T> plane(model_n * board_size * board_size, 0);

        // id-th plane を 1 で埋める
        const int offset = id * board_size * board_size;
        for (int i = 0; i < board_size * board_size; ++i) {
            plane[offset + i] = 1;
        }

        return plane;
    }

}  // namespace feature_impl
}  // namespace cygo


template <typename Zone>
void cygo::feature_impl::batch_features_with_zone_to_ptr(
    std::vector<State> const& state_list, int history_n,
    Zone zone_seq,  // idx -> pointer to const int8[channel_size]
    int channel_size, int size_per_state, int8_t* data, int8_t* legals_out) {

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
            auto zone = zone_seq(i);
            std::copy(&zone[0], &zone[channel_size], &data[zone_plane_offset]);

            const auto& stones_b = state.black_board();
            const auto& stones_w = state.white_board();
            const int base = i * (channel_size + 1);
            for (size_t j = 0; j < stones_b.size(); ++j) {
                int id = base + j;
                assert (0 <= id);
                legals_out[id] = 1 - (stones_b[j] + stones_w[j]);
            }

            legals_out[base + channel_size] = 1;
        }
    };
    task(0, state_list.size());
}


// todo factor out this block?
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

#endif //CYGO_IMPL_FEATURES_HPP
