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
std::vector<T> board_i_color(State const& state, int i, Color c) {
    // Returns the color's plane of S at T_{t-i}.
    // The shape of the resultant array is (1, size, size)
    assert(c != Color::EMPTY);

    auto const& history = state.history().history(c);

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
std::vector<T> board_i(State const& state, int i) {
    assert(0 <= i and i < state.history().history_length);

    // Returns the plane of S at T_{t-i}
    // shape: (2, size, size)
    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * 2;

    std::vector<T> ret(array_size, T(0));

    auto const& history_1 = state.history().history(state.current_player);
    auto const& history_2 = state.history().history(opposite_color(state.current_player));

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
std::vector<T> history_n_color(State const& state, int n, Color c) {
    assert(0 <= n and n < state.history().history_length);

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * (n + 1);

    std::vector<T> ret(array_size, T(0));

    auto const& history = state.history().history(c);
    auto itr = std::begin(history);

    int i = 0;
    int length = history.size();

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

template <typename T>
std::vector<T> history_n(State const& state, int n) {
    assert(0 <= n and n < state.history().history_length);

    auto board_size = static_cast<std::size_t>(state.board_size());
    auto plane_size = board_size * board_size;
    auto array_size = plane_size * 2 * (n + 1);

    std::vector<T> ret(array_size, T(0));

    auto const& history_1 = state.history().history(state.current_player);
    auto const& history_2 = state.history().history(opposite_color(state.current_player));

    auto itr_1 = std::begin(history_1);
    auto itr_2 = std::begin(history_2);

    int i = 0;
    int length = history_1.size();

    auto ret_itr = std::begin(ret);

    while (i <= n and i < length) {
        auto const& board_1 = *itr_1;
        auto const& board_2 = *itr_2;

        std::copy(std::begin(board_1), std::end(board_1), ret_itr);
        std::copy(std::begin(board_2), std::end(board_2), std::next(ret_itr, plane_size));

        ++itr_1; ++itr_2;
        std::advance(ret_itr, 2 * plane_size);
        i += 1;
    }

    return ret;
}

}  // namespace feature_impl
}  // namespace cygo

#endif //CYGO_IMPL_FEATURES_HPP
