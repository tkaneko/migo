#include <gtest/gtest.h>

#include "cygo/chain_group.hpp"

using namespace cygo;


TEST(chain_group_test, place_stone_merge_chain) {
    int board_size = 9;

    ChainGroup g(board_size);

    for (auto const& v : {Move::from_coordinate(4, 4, board_size),
                          Move::from_coordinate(5, 4, board_size),
                          Move::from_coordinate(3, 4, board_size)}) {
        g.place_stone(Color::BLACK, v);
    }

    auto chain = g.chain_at(Move::from_coordinate(4, 4, board_size));

    ASSERT_EQ(8, chain.liberty_count());
}


TEST(chain_group_test, place_stone_remove_chain) {
    int board_size = 9;

    ChainGroup g(board_size);

    Move white = Move::from_coordinate(4, 4, board_size);

    g.place_stone(Color::WHITE, white);

    ASSERT_EQ(4, g.chain_at(white).liberty_count());
    ASSERT_EQ(Color::WHITE, g.stone_at(white));
    ASSERT_FALSE(g.chain_at(white).is_in_atari());

    for (auto const &v : {Move::from_coordinate(4, 5, board_size),
                          Move::from_coordinate(5, 4, board_size),
                          Move::from_coordinate(4, 3, board_size),
                          Move::from_coordinate(3, 4, board_size)}) {
        g.place_stone(Color::BLACK, v);
    }

    ASSERT_EQ(Color::EMPTY, g.stone_at(white));
    ASSERT_EQ(0, g.empty_count_around(white));
    ASSERT_TRUE(g.is_eye_like(Color::BLACK, white));
    ASSERT_FALSE(g.has_chain_at(white));
}

TEST(chain_group_test, place_stone_remove_multiple_stones) {
    int board_size = 9;

    ChainGroup g(board_size);

    g.place_stone(Color::WHITE, Move::from_coordinate(4, 4, board_size));
    g.place_stone(Color::WHITE, Move::from_coordinate(5, 4, board_size));

    ASSERT_EQ(2, g.chain_at(Move::from_coordinate(4, 4, board_size)).size());
    ASSERT_EQ(6, g.chain_at(Move::from_coordinate(4, 4, board_size)).liberty_count());

    for (auto const &v : {Move::from_coordinate(4, 3, board_size),
                          Move::from_coordinate(5, 3, board_size),
                          Move::from_coordinate(4, 5, board_size),
                          Move::from_coordinate(5, 5, board_size),
                          Move::from_coordinate(6, 4, board_size),
                          Move::from_coordinate(3, 4, board_size)}) {
        g.place_stone(Color::BLACK, v);
    }

    ASSERT_FALSE(g.has_chain_at(Move::from_coordinate(4, 4, board_size)));

    ASSERT_EQ(Color::EMPTY, g.stone_at(Move::from_coordinate(4, 4, board_size)));
    ASSERT_EQ(Color::EMPTY, g.stone_at(Move::from_coordinate(5, 4, board_size)));
}