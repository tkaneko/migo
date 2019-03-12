#include <gtest/gtest.h>

#include "cygo/move.hpp"

using namespace cygo;


TEST(vertex_test, check_any) {
    ASSERT_NE(Move::ANY, Move::PASS);
}

TEST(vertex_test, from_coordinate) {
    int board_size = 19;

    ASSERT_EQ(Move::from_coordinate( 0,  0, board_size).raw(),   0);
    ASSERT_EQ(Move::from_coordinate( 0, 18, board_size).raw(),  18);
    ASSERT_EQ(Move::from_coordinate(18, 18, board_size).raw(), 360);
}

TEST(vertex_test, from_coordinate_invalid_coords) {
    ASSERT_THROW(Move::from_coordinate(-1,  0, 19), std::invalid_argument);
    ASSERT_THROW(Move::from_coordinate( 0, 19, 19), std::invalid_argument);
    ASSERT_THROW(Move::from_coordinate(19, 19, 19), std::invalid_argument);
}

TEST(vertex_test, from_gtp_string) {
    int board_size = 19;

    ASSERT_EQ(Move::from_coordinate(0, 0, board_size), Move::from_gtp_string("A1", board_size));
    ASSERT_EQ(Move::from_coordinate(0, 0, board_size), Move::from_gtp_string("a1", board_size));

    ASSERT_EQ(Move::PASS, Move::from_gtp_string("PASS", board_size));
    ASSERT_EQ(Move::PASS, Move::from_gtp_string("pass", board_size));
}

TEST(vertex_test, from_gtp_string_invalid_coords) {
    int board_size = 9;

    ASSERT_THROW(Move::from_gtp_string("aaaaa", board_size), std::invalid_argument);
    ASSERT_THROW(Move::from_gtp_string("A111", board_size), std::invalid_argument);
    ASSERT_THROW(Move::from_gtp_string("AA1", board_size), std::invalid_argument);
    ASSERT_THROW(Move::from_gtp_string("K1", board_size), std::invalid_argument);
}

TEST(vertex_test, for_each_4nbr) {
    Move v(Move::from_coordinate(0, 0, 19));

    std::size_t expected_nbrs = 2, actual_nbrs = 0;

    for_each_4nbr(v, [&] (Move const& nbr) {
        actual_nbrs += 1;
    });

    ASSERT_EQ(expected_nbrs, actual_nbrs);
}