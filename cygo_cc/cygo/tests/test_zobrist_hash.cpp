#include <gtest/gtest.h>

#include "cygo/zobrist_hash.hpp"

using namespace cygo;


TEST(zobrist_hash_test, twice) {
    ZobristHash hash;

    auto old_value = hash.hash_value();

    hash.update(Color::BLACK, Move::from_coordinate(3, 3, 19));

    ASSERT_NE(old_value, hash.hash_value());

    hash.update(Color::BLACK, Move::from_coordinate(3, 3, 19));

    ASSERT_EQ(old_value, hash.hash_value());
}

TEST(zobrist_hash_test, move_order) {
    int size = 19;

    auto hash1 = ZobristHash::calculate_hash({Move::from_gtp_string("E5", size), Move::from_gtp_string("F1", size)}, {});
    auto hash2 = ZobristHash::calculate_hash({Move::from_gtp_string("F1", size), Move::from_gtp_string("E5", size)}, {});

    ASSERT_EQ(hash1, hash2);
}
