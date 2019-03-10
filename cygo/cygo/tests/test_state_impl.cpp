#include <gtest/gtest.h>

#include "cygo/state_impl.hpp"

using namespace cygo;


TEST(state_impl_test, is_legal_initial_state) {
    for (int i = 1; i <= 25; ++i) {
        StateImpl s(i, false);

        ASSERT_EQ(i * i, s.legal_moves(Color::BLACK, false).size());
    }
}


TEST(state_impl_test, is_legal_suicide_move) {
    {
        StateImpl s(9, false);

        s.make_move(Color::BLACK, Move::from_gtp_string("F5", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("E4", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("E6", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("D5", 9));

        ASSERT_TRUE(s.is_legal(Color::BLACK, Move::from_gtp_string("E5", 9)));
        ASSERT_FALSE(s.is_legal(Color::WHITE, Move::from_gtp_string("E5", 9)));
    }

    {
        StateImpl s(9, false);

//          A B C D E F G H J
//        9 . . . . . . . . . 9
//        8 . . . . . . . . . 8
//        7 . . + . + . + . . 7
//        6 . . . O X . . . . 6
//        5 . . O . O X + . . 5
//        4 . . O X X O . . . 4
//        3 . . + O O . + . . 3
//        2 . . . . . . . . . 2
//        1 . . . . . . . . . 1
//          A B C D E F G H J

        s.make_move(Color::BLACK, Move::from_gtp_string("E6", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("F5", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("E4", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("D4", 9));

        s.make_move(Color::WHITE, Move::from_gtp_string("D6", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("C5", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("E5", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("C4", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("F4", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("D3", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("E3", 9));

        ASSERT_TRUE(s.is_legal(Color::BLACK, Move::from_gtp_string("D5", 9)));
        ASSERT_TRUE(s.is_legal(Color::WHITE, Move::from_gtp_string("D5", 9)));
    }
}

TEST(state_impl_test, is_legal_ko) {
    {
        StateImpl s(9, false);

        s.make_move(Color::BLACK, Move::from_gtp_string("B8", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("A8", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("C9", 9));
        s.make_move(Color::WHITE, Move::from_gtp_string("B9", 9));
        s.make_move(Color::BLACK, Move::from_gtp_string("A9", 9));

        ASSERT_FALSE(s.is_legal(Color::WHITE, Move::from_gtp_string("B9", 9)));
    }
}

