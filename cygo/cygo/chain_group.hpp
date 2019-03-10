#ifndef CYGO_CHAIN_GROUP_HPP
#define CYGO_CHAIN_GROUP_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chain.hpp"
#include "color.hpp"
#include "neighbor_counter.hpp"
#include "move.hpp"
#include "zobrist_hash.hpp"

namespace cygo {

class ChainGroup {
public:
    explicit ChainGroup(int board_size);

    void place_stone(Color color, Move const& vertex);

    Chain const& chain_at(Move const& v) const;
    Color stone_at(Move const& v) const;

    std::unordered_set<Move> const& empties() const;

    std::vector<int> const& black_board() const;
    std::vector<int> const& white_board() const;

    std::vector<Color> const& stones() const;

    int count_stones(Color color) const;
    int empty_count_around(Move const& v) const;

    bool has_chain_at(Move const& v) const;
    bool is_atari_group(Move const& v) const;
    bool is_eye_like(Color color, Move const& vertex) const;

    ZobristHash::ValueType hash() const;

    std::string to_string() const;

private:
    Chain& chain_at_(Move const& v, bool with_check = true);

    void set_stone(Color c, Move const& v);

    void merge_chains(Move const& v1, Move const& v2);
    void remove_chain(Move const& v);
    void remove_stone(Move const& v);

private:
    const int board_size_;

    ZobristHash hash_;

    std::vector<Color> stones_;
    std::vector<int> black_stones_;
    std::vector<int> white_stones_;

    std::vector<int> chain_ids_;
    std::vector<NeighborCounter> neighbor_counters_;

    std::unordered_set<Move> empties_;
    std::unordered_map<int, Chain> chains_;
};

}  // namespace  cygo

#endif //CYGO_CHAIN_GROUP_HPP
