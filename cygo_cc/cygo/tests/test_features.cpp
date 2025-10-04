#include <gtest/gtest.h>

#include "cygo/features.hpp"

using namespace cygo;
using namespace cygo::feature_impl;


TEST(features_test, color_impl_when_current_is_black) {
    std::size_t size = 9;

    State s(size, 7.5, false, 10);

    ASSERT_EQ(Color::BLACK, s.current_player);

    std::vector<float> expected_black(size * size, 1.0f);
    std::vector<float> expected_white(size * size, 0.0f);

    ASSERT_EQ(expected_black, color<float>(s, Color::BLACK));
    ASSERT_EQ(expected_white, color<float>(s, Color::WHITE));
}

TEST(features_test, color_impl_when_current_is_white) {
    std::size_t size = 9;

    State s(size, 7.5, false, 10);

    s.current_player = Color::WHITE;

    ASSERT_EQ(Color::WHITE, s.current_player);

    std::vector<float> expected_black(size * size, 0.0f);
    std::vector<float> expected_white(size * size, 1.0f);

    ASSERT_EQ(expected_black, color<float>(s, Color::BLACK));
    ASSERT_EQ(expected_white, color<float>(s, Color::WHITE));
}

TEST(features_test, board_i_color_impl) {
    std::size_t size = 9;

    State s(size, 7.5, false, 10);

    ASSERT_EQ(std::vector<float>(size * size, 0.0f), board_i_color<float>(s, 0, Color::BLACK));
    ASSERT_EQ(std::vector<float>(size * size, 0.0f), board_i_color<float>(s, 0, Color::WHITE));

    s.make_move(Move::from_raw(0, size));

    std::vector<float> expected_black = {
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    ASSERT_EQ(expected_black, board_i_color<float>(s, 0, Color::BLACK));
}

TEST(features_test, game_move_pair) {
    const std::vector<int> game_index = {0, 89, 89, 200};
    const int idx[] = {3, 88, 89, 90, 200};
    const int gid[] = {0,  0,  2,  2,   3};
    const int mid[] = {3, 88,  0,  1,   0};

    for (int i=0; i<5; ++i) {
        auto [g, m] = to_game_move_pair(game_index, idx[i], 0);
        ASSERT_EQ(gid[i], g);
        ASSERT_EQ(mid[i], m);
    }
}
