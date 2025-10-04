#ifndef CYGO_STATE_IMPL_HPP
#define CYGO_STATE_IMPL_HPP

#include "chain_group.hpp"
#include "color.hpp"
#include "neighbor_counter.hpp"
#include "move.hpp"
#include "zobrist_hash.hpp"

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

namespace cygo {

class StateImpl {
public:
    StateImpl(int board_size, bool superko_rule);

    void make_move(Color c, Move const& v);
    void drop_history();

    std::unordered_set<Move> legal_moves(Color c, bool include_eye_likes) const;

    bool is_eye_like(Color c, Move const &v) const;
    bool is_legal(Color c, Move const& v, std::string *msg=nullptr) const;
    bool is_suicide_move(Color c, Move const& v) const;
    bool is_positional_superko(Color c, Move const& v, std::string *err=nullptr) const;

    double tromp_taylor_score(Color c, int8_t *board_zero_filled=nullptr) const;

    int board_size() const;
    bool superko_rule() const;

    ZobristHash::ValueType hash() const;

    ChainGroup const& chain_group() const;

    std::vector<Move> move_history(Color c) const;
    std::string info() const;
private:
    const int board_size_;
    bool superko_rule_;

    ChainGroup chain_group_;

    Move ko_vertex_;

    PlayerMap<std::vector<Move>> color_move_history_;
    std::unordered_map<ZobristHash::ValueType, int> hash_history_;
    std::vector<Move> move_history_;
};

}  // namespace cygo

#endif //CYGO_STATE_IMPL_HPP
