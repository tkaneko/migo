#include <gtest/gtest.h>

#include "cygo/move.hpp"
#include "cygo/state.hpp"

using namespace cygo;


TEST(state_test, real_game_1) {
    State s(9, 7.5, false, 0);

    auto moves = {
            "D5", "C3", "E7", "F3", "G5", "E4", "C4", "B3", "G3", "G4",
            "H4", "F4", "H2", "G2", "H3", "F6", "F5", "E5", "E6", "F7",
            "G6", "E8", "D8", "C9", "F8", "D6", "D7", "C6", "C7", "B6",
            "B7", "A7", "B5", "G7", "C5", "H7", "H6", "G8", "E9", "F2",
            "J5", "J6", "J7", "J8", "H9", "J6", "H5", "G9", "H1", "J2",
            "J3", "F9", "E8", "A6", "A5", "A8", "B8", "B4", "A9", "D4",
            "A4", "D6", "A3", "B2", "A2", "A1", "E3", "G1", "D2", "D3",
            "J7", "B1", "H8", "B6", "C6", "E2", "A6", "H7", "G7", "A8",
            "A7", "B9", "J9", "A8", "J1", "A9", "C8", "D1", "D9", "C2",
            "E1", "F1", "F6", "G9", "F9", "C9", "B9", "A9", "A8", "PASS",
            "G8", "PASS", "PASS"
    };

    /*
         A  B  C  D  E  F  G  H  J
       9 .  X  .  X  X  X  .  X  X 9
       8 X  X  X  X  X  X  X  X  . 8
       7 X  X  X  X  X  .  X  .  X 7
       6 X  .  X  .  X  X  X  X  . 6
       5 X  X  X  X  O  X  X  X  X 5
       4 X  O  X  O  O  O  O  X  . 4
       3 X  O  O  O  .  O  X  X  X 3
       2 X  O  O  .  O  O  O  X  . 2
       1 O  O  .  O  .  O  O  X  X 1
         A  B  C  D  E  F  G  H  J
    */

    for (auto m : moves) {
        s.make_move(Move::from_gtp_string(m, 9));
    }

    std::vector<uint8_t> black_board = {
            0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 0, 1, 0, 0, 0, 0, 1, 0,
            1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 0, 1, 0, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 1, 0, 1, 1,
    };

    std::vector<uint8_t> white_board = {
            1, 1, 0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0, 0, 0,
            0, 1, 0, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    ASSERT_EQ(black_board, s.black_board());
    ASSERT_EQ(white_board, s.white_board());
}


TEST(state_test, real_game_2) {
    State s(9, 7.5, false, 0);

    auto moves = {
            "F6", "D7", "E7", "D6", "F4", "D4", "D3", "C3", "E3", "G5",
            "G6", "F5", "E5", "E4", "G4", "H5", "D5", "C4", "E6", "C5",
            "H4", "F3", "F2", "E8", "F8", "D8", "C2", "H6", "G7", "B2",
            "C6", "H8", "G8", "C7", "B6", "H7", "H9", "E9", "J5", "D2",
            "E2", "F9", "G9", "C1", "J6", "B7", "G3", "E1", "F1", "D1",
            "J8", "G2", "J7", "G1", "F3", "H2", "H3", "J3", "J4", "J1",
            "J2", "G5", "H1", "B5", "B1", "A1", "A2", "A3", "B3", "A6",
            "A4", "B4", "J9", "A5", "J1", "D9", "B8", "B9", "A8", "A7",
            "C8", "C9", "C6", "A9", "B8", "A8", "H5", "H7", "H6", "G2",
            "G1", "A2", "H2", "B6", "F5", "C8", "H8", "PASS", "PASS"
    };

    /*
        A  B  C  D  E  F  G  H  J
      9 O  O  O  O  O  O  X  X  X 9
      8 O  .  O  O  O  X  X  X  X 8
      7 O  O  O  O  X  .  X  .  X 7
      6 O  O  .  O  X  X  X  X  X 6
      5 O  O  O  X  X  X  .  X  X 5
      4 .  O  O  O  O  X  X  X  X 4
      3 O  .  O  X  X  X  X  X  . 3
      2 O  O  .  O  X  X  .  X  X 2
      1 O  .  O  O  O  X  X  X  X 1
        A  B  C  D  E  F  G  H  J
    */

    for (auto const &m : moves) {
        s.make_move(Move::from_gtp_string(m, 9));
    }

    std::vector<uint8_t> black_board = {
            0, 0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 0, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 1, 1, 1, 0, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 1, 1, 1,
    };

    std::vector<uint8_t> white_board = {
            1, 0, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0, 0,
    };

    ASSERT_EQ(black_board, s.black_board());
    ASSERT_EQ(white_board, s.white_board());
}

TEST(state_test, real_game_3) {

    auto moves = {
            "D4", "E7", "G6", "E5", "D5", "E4", "E6", "F6", "D6", "F7",
            "F5", "E3", "G4", "G7", "D3", "D2", "C2", "G2", "H3", "E2",
            "H6", "D7", "C7", "C3", "B3", "F4", "G5", "H2", "H7", "C8",
            "B7", "H8", "J8", "H9", "B8", "D9", "J5", "J7", "J2", "G3",
            "J3", "H4", "D1", "E1", "F2", "B2", "C4", "C1", "H1", "C3",
            "G1", "B4", "C2", "A3", "F9", "F3", "F8", "C3", "B5", "A5",
            "C6", "A7", "A8", "E8", "B9", "G8", "E9", "F1", "A6", "J1",
            "H1", "G1", "J6", "G9", "C9", "F9", "D8", "J1", "E9", "J4",
            "A4", "B3", "D9", "H5", "A5", "J7", "H7", "H6", "J6", "J5",
            "B1", "A1", "J2", "H1", "G4", "J9", "F5", "G5", "J3", "H3",
            "J2", "J3", "PASS", "PASS"
    };

    State s(9, 7.5, false, 0);

    for (auto m : moves) {
        s.make_move(Move::from_gtp_string(m, 9));
    }

    /*
        A B C D E F G H J
      9 . X X X X O O O O 9
      8 X X . X O . O O . 8
      7 . X X O O O O . O 7
      6 X . X X X O . O . 6
      5 X X . X O . O O O 5
      4 X O X X O O . O O 4
      3 O O O X O O O O O 3
      2 . O . O O . O O . 2
      1 O . O . O O O O O 1
        A B C D E F G H J
     */

    std::vector<uint8_t> black_board = {
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 1, 0, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 1, 1, 1, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 0, 0,
    };

    std::vector<uint8_t> white_board = {
            1, 0, 1, 0, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 1, 0, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 1, 1, 0, 1, 1,
            0, 0, 0, 0, 1, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 0, 1, 1, 1, 1, 0, 1,
            0, 0, 0, 0, 1, 0, 1, 1, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 1,
    };

    ASSERT_EQ(black_board, s.black_board());
    ASSERT_EQ(white_board, s.white_board());
}

TEST(state_history_size_test, history_size) {
    const auto history_n = 3;
    State s(9, 7.5, false, history_n);
    ASSERT_TRUE(s.max_history_n() == history_n);
    ASSERT_TRUE(s.history(Color::BLACK).size() == 0);
    ASSERT_TRUE(s.history(Color::WHITE).size() == 0);
    for (auto move: {"D5", "C3", "E7"}) {
        s.make_move(Move::from_gtp_string(move, 9));
    }
    ASSERT_TRUE(s.history(Color::BLACK).size() == history_n);
    ASSERT_TRUE(s.history(Color::WHITE).size() == history_n);

    s.make_move(Move::from_gtp_string("F3", 9));
    ASSERT_TRUE(s.history(Color::BLACK).size() == history_n+1);
    ASSERT_TRUE(s.history(Color::WHITE).size() == history_n+1);

    s.make_move(Move::from_gtp_string("G5", 9));
    ASSERT_TRUE(s.history(Color::BLACK).size() == history_n+1);
    ASSERT_TRUE(s.history(Color::WHITE).size() == history_n+1);
}


TEST(state_test, drop_history) {
    State s(9, 7.5, false, 0);

    auto moves = {
            "D5", "C3", "E7", "F3", "G5", "E4", "C4", "B3", "G3", "G4",
            "H4", "F4", "H2", "G2", "H3", "F6", "F5", "E5", "E6", "F7",
            "G6", "E8", "D8", "C9", "F8", "D6", "D7", "C6", "C7", "B6",
    };

    for (auto m : moves) {
        s.make_move(Move::from_gtp_string(m, 9));
    }

    auto black_board = s.black_board();
    auto white_board = s.white_board();

    ASSERT_EQ(s.last_move(), Move::from_gtp_string(*std::rbegin(moves), 9));
    ASSERT_TRUE(s.history(Color::BLACK).size() > 0);
    ASSERT_TRUE(s.history(Color::WHITE).size() > 0);
    
    s.drop_history();

    ASSERT_EQ(s.last_move(), Move::INVALID);

    ASSERT_EQ(black_board, s.black_board());
    ASSERT_EQ(white_board, s.white_board());

    ASSERT_TRUE(s.history(Color::BLACK).size() == 1);
    ASSERT_TRUE(s.history(Color::WHITE).size() == 1);
}

TEST(state_test, score) {
    State s(9, 7.5, false, 0);
    std::vector<int8_t> board(9*9u, 0);
    ASSERT_EQ(s.tromp_taylor_score(Color::EMPTY, &*board.begin()), -7.5);
    ASSERT_EQ(0, *std::max_element(board.begin(), board.end()));
    ASSERT_EQ(0, *std::min_element(board.begin(), board.end()));
    
    s.make_move(Move::from_gtp_string("E5", 9));
    ASSERT_EQ(s.tromp_taylor_score(Color::EMPTY, &*board.begin()), -73.5);

    ASSERT_EQ(1, *std::max_element(board.begin(), board.end()));
    ASSERT_EQ(1, *std::min_element(board.begin(), board.end()));
}
