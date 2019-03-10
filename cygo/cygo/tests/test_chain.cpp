#include <gtest/gtest.h>

#include "cygo/chain.hpp"

using namespace cygo;


TEST(chain_test, default_constructor) {
    Chain chain;

    ASSERT_TRUE(chain.members().empty());
    ASSERT_EQ(Move::INVALID, chain.atari_vertex());
}


TEST(chain_test, constructor) {
    {
        Chain chain(Move::from_coordinate(4, 4, 9));

        ASSERT_EQ(4, chain.liberty_count());
        ASSERT_EQ(1, chain.size());
        ASSERT_FALSE(chain.is_in_atari());
        ASSERT_FALSE(chain.is_captured());
    }

    {
        Chain chain(Move::from_coordinate(0, 1, 9));

        ASSERT_EQ(3, chain.liberty_count());
        ASSERT_EQ(1, chain.size());
        ASSERT_FALSE(chain.is_in_atari());
        ASSERT_FALSE(chain.is_captured());
    }

    {
        Chain chain(Move::from_coordinate(0, 0, 9));

        ASSERT_EQ(2, chain.liberty_count());
        ASSERT_EQ(1, chain.size());
        ASSERT_FALSE(chain.is_in_atari());
        ASSERT_FALSE(chain.is_captured());
    }
}


TEST(chain_test, is_in_atari) {
    int board_size = 9;

    Move v(Move::from_coordinate(4, 4, board_size));

    Chain chain(v);

    for (auto const& v : {Move::from_coordinate(3, 4, board_size),
                          Move::from_coordinate(5, 4, board_size),
                          Move::from_coordinate(4, 5, board_size)}) {
        chain.add_adjacent_opponent(v);
    }

    ASSERT_TRUE(chain.is_in_atari());
    ASSERT_EQ(Move::from_coordinate(4, 3, board_size), chain.atari_vertex());
}

