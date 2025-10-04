#ifndef CYGO_IMPL_FEATURES_HPP
#define CYGO_IMPL_FEATURES_HPP

#include <cassert>

#include "color.hpp"
#include "state.hpp"


namespace cygo {
namespace feature_impl {

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

}  // namespace feature_impl
}  // namespace cygo

#endif //CYGO_IMPL_FEATURES_HPP
