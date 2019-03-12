#include <gtest/gtest.h>

#include "cygo/neighbor_counter.hpp"

using namespace cygo;


TEST(neighbor_counter_test, creation) {
    NeighborCounter n;

    ASSERT_EQ(0, n.player_count(Color::BLACK));
    ASSERT_EQ(0, n.player_count(Color::WHITE));
    ASSERT_EQ(4, n.empty_count());
}

TEST(neighbor_counter_test, increment) {
    NeighborCounter n;

    n.increment(Color::BLACK);

    ASSERT_EQ(1, n.player_count(Color::BLACK));
    ASSERT_EQ(0, n.player_count(Color::WHITE));
    ASSERT_EQ(3, n.empty_count());

    n.increment(Color::WHITE);

    ASSERT_EQ(1, n.player_count(Color::BLACK));
    ASSERT_EQ(1, n.player_count(Color::WHITE));
    ASSERT_EQ(2, n.empty_count());

    n.increment(Color::BLACK);
    n.increment(Color::WHITE);

    ASSERT_EQ(2, n.player_count(Color::BLACK));
    ASSERT_EQ(2, n.player_count(Color::WHITE));
    ASSERT_EQ(0, n.empty_count());
}

TEST(neighbor_counter_test, decrement) {
    NeighborCounter n;

    n.increment(Color::BLACK);
    n.increment(Color::BLACK);
    n.increment(Color::WHITE);
    n.increment(Color::WHITE);

    ASSERT_EQ(2, n.player_count(Color::BLACK));
    ASSERT_EQ(2, n.player_count(Color::WHITE));
    ASSERT_EQ(0, n.empty_count());

    n.decrement(Color::BLACK);

    ASSERT_EQ(1, n.player_count(Color::BLACK));
    ASSERT_EQ(2, n.player_count(Color::WHITE));
    ASSERT_EQ(1, n.empty_count());

    n.decrement(Color::WHITE);

    ASSERT_EQ(1, n.player_count(Color::BLACK));
    ASSERT_EQ(1, n.player_count(Color::WHITE));
    ASSERT_EQ(2, n.empty_count());
}