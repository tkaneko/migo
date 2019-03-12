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