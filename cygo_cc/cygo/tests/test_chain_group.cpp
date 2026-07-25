#include <gtest/gtest.h>

#include "cygo/chain_group.hpp"

using namespace cygo;

TEST(chain_group_test, uft) {
    const int N = 10;
    UFT tree(N);
    for (int i = 0; i < N; ++i) {
        ASSERT_TRUE(tree.empty(i));
        ASSERT_TRUE(tree.root(i) == i);
    }

    int id = 3;
    tree.enable(id);
    ASSERT_FALSE(tree.empty(id));
    ASSERT_TRUE(tree.root(id) == id);
    ASSERT_TRUE(tree.size(id) == 1);

    int id5 = 5;
    tree.enable(id5);
    ASSERT_FALSE(tree.empty(id5));
    ASSERT_TRUE(tree.root(id5) == id5);
    ASSERT_TRUE(tree.size(id5) == 1);

    ASSERT_TRUE(tree.unite(id5, id5) == 0);

    ASSERT_TRUE(tree.unite(id, id5) > 0);
    ASSERT_TRUE(tree.root(id) == id || tree.root(id) == id5);
    ASSERT_TRUE(tree.root(id5) == id || tree.root(id5) == id5);
    ASSERT_TRUE(tree.size(tree.root(id)) == 2);
    ASSERT_TRUE(tree.size(tree.root(id5)) == 2);
}


TEST(vertex_list_test, vlist) {
    const int N = 10;
    VertexList vl(N);
    
    for (int i = 0; i < N; ++i) {
        ASSERT_FALSE(vl.next(i));
        // ASSERT_FALSE(vl.prev(i));
    }
}

TEST(chain_group_test, place_stone_merge_chain) {
    int board_size = 9;

    ChainGroup g(board_size);

    auto stones = {Move::from_coordinate(4, 4, board_size),
                          Move::from_coordinate(5, 4, board_size),
                          Move::from_coordinate(3, 4, board_size)};
    for (auto const& v : stones) {
        g.place_stone(Color::BLACK, v);
    }

    auto chain = g.chain_at(Move::from_coordinate(4, 4, board_size));

    ASSERT_EQ(8, chain.liberty_count());

    for (auto v: stones) {
        ASSERT_EQ(3, g.chain_members_at(v).size());
    }
}

TEST(chain_group_test, atari_chain) {
    int board_size = 9;

    ChainGroup g(board_size);

    auto stones
        = { Move::from_coordinate(4, 3, board_size),
            Move::from_coordinate(3, 4, board_size),
            Move::from_coordinate(4, 4, board_size)
            };
    for (auto const& v : stones) {
        g.place_stone(Color::BLACK, v);
    }

    auto chain = g.chain_at(Move::from_coordinate(4, 4, board_size));

    ASSERT_EQ(8, chain.liberty_count()); // actually 7 for human
    for (auto v: stones) {
        ASSERT_EQ(3, g.chain_members_at(v).size());
    }

    auto stones_w
        = { Move::from_coordinate(2, 4, board_size),
            Move::from_coordinate(3, 3, board_size),
            Move::from_coordinate(3, 5, board_size),
            Move::from_coordinate(4, 2, board_size),
            Move::from_coordinate(4, 5, board_size),
            Move::from_coordinate(5, 3, board_size),
            };
    for (auto const& v : stones_w) {
        g.place_stone(Color::WHITE, v);
    }
    auto chain2 = g.chain_at(Move::from_coordinate(4, 4, board_size));
    ASSERT_EQ(1, chain2.liberty_count());
    ASSERT_TRUE(chain2.is_in_atari());
    ASSERT_EQ(Move::from_coordinate(5, 4, board_size), chain2.atari_vertex(board_size));
}

TEST(chain_group_test, atari_chain_corner) {
    int board_size = 9;

    ChainGroup g(board_size);

    auto stones
        = { Move::from_coordinate(4, 3, board_size),
            Move::from_coordinate(3, 4, board_size),
            Move::from_coordinate(4, 4, board_size)
            };
    for (auto const& v : stones) {
        g.place_stone(Color::BLACK, v);
    }

    auto chain = g.chain_at(Move::from_coordinate(4, 4, board_size));

    ASSERT_EQ(8, chain.liberty_count()); // actually 7 for human
    for (auto v: stones) {
        ASSERT_EQ(3, g.chain_members_at(v).size());
    }

    auto stones_w
        = { Move::from_coordinate(2, 4, board_size),
            Move::from_coordinate(3, 5, board_size),
            Move::from_coordinate(4, 2, board_size),
            Move::from_coordinate(4, 5, board_size),
            Move::from_coordinate(5, 3, board_size),
            Move::from_coordinate(5, 4, board_size),
            };
    for (auto const& v : stones_w) {
        g.place_stone(Color::WHITE, v);
    }
    auto chain2 = g.chain_at(Move::from_coordinate(4, 4, board_size));
    ASSERT_EQ(2, chain2.liberty_count()); // 1 for human
    ASSERT_TRUE(chain2.is_in_atari());
    ASSERT_EQ(Move::from_coordinate(3, 3, board_size), chain2.atari_vertex(board_size));
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

    ASSERT_EQ(2, g.chain_size_at(Move::from_coordinate(4, 4, board_size)));
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
