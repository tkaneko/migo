#include "chain_group.hpp"

#include <cassert>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

namespace cygo {

ChainGroup::ChainGroup(int board_size)
    : board_size_(board_size),
      stones_(board_size * board_size, Color::EMPTY),
      black_stones_(board_size * board_size, 0),
      white_stones_(board_size * board_size, 0),
      chain_tree_(board_size * board_size),
      chain_list_(board_size * board_size),
      neighbor_counters_(board_size * board_size),
      chains_(board_size * board_size)
{
    for_each_coordinate(board_size_, [&](Move const& v) {

        if (v.is_at_corner()) {
            neighbor_counters_[v()] = NeighborCounter::empty_at_corner();
        } else if (v.is_on_edge()) {
            neighbor_counters_[v()] = NeighborCounter::empty_on_edge();
        } else {
            neighbor_counters_[v()] = NeighborCounter::empty();
        }
    });
}

void ChainGroup::place_stone(Color c, Move const& v) {
    hash_.update(c, v);

    set_stone_(c, v);

    auto id = chain_id(v);
    chains_[id].emplace(v, id); // chain's head and tail are initialized here

    for_each_4nbr(v, [&] (Move const& nbr) {
        neighbor_counters_[nbr()].increment(c);
    });

    // update liberty counts first
    for_each_4nbr(v, [&] (Move const& nbr) {
        Color nbr_color = stone_at(nbr);

        if (nbr_color == Color::EMPTY) {
            return;             // i.e., continue
        }
        else if (nbr_color == c) {
            chain_at_(nbr).add_adjacent_ally(v);
            chain_at_(v).add_adjacent_ally(nbr);
        }
        else {
            chain_at_(nbr).add_adjacent_opponent(v);
            chain_at_(v).add_adjacent_opponent(nbr);
        }
    });

    // and then remove/merge if necessary
    for_each_4nbr(v, [&] (Move const& nbr) {
        Color nbr_color = stone_at(nbr);

        if (nbr_color == Color::EMPTY) {
            return;             // i.e., continue
        }

        if (nbr_color == c) {
            merge_chains_(v, nbr);
        }
        else if (chain_at(nbr).is_captured()){
            remove_chain_(nbr);
        }
    });
}

Chain const& ChainGroup::chain_at(Move const& v) const {    
    return *chains_.at(chain_id(v));
}

Color ChainGroup::stone_at(Move const& v) const {
    // NAND( black, white ) should be true
    assert(black_stones_[v()] * white_stones_[v()] == 0);

    if (black_stones_[v()] != 0) {
        return Color::BLACK;
    }

    if (white_stones_[v()] != 0) {
        return Color::WHITE;
    }

    return Color::EMPTY;
}

int ChainGroup::count_stones(Color color) const {
    if (color == Color::BLACK) {
        return std::accumulate(std::begin(black_stones_), std::end(black_stones_), 0);
    }

    if (color == Color::WHITE) {
        return std::accumulate(std::begin(white_stones_), std::end(white_stones_), 0);
    }

    return 0;
}

int ChainGroup::empty_count_around(Move const& v) const {
    return neighbor_counters_[v()].empty_count();
}

bool ChainGroup::is_atari_group(Move const& v) const {
    return chain_at(v).is_in_atari();
}

bool ChainGroup::is_eye_like(Color c, Move const& v) const {
    return neighbor_counters_[v()].is_eye_like(c);
}

bool ChainGroup::has_chain_at(Move const& v) const {
    return chain_id_or_empty(v).has_value();
}

ZobristHash::ValueType ChainGroup::hash() const {
    return hash_.hash_value();
}

std::vector<uint8_t> const& ChainGroup::black_board() const {
    return black_stones_;
}

std::vector<uint8_t> const& ChainGroup::white_board() const {
    return white_stones_;
}

std::vector<Color> const& ChainGroup::stones() const {
    return stones_;
}

Chain& ChainGroup::chain_at_(Move const& v) {
    auto& chain_opt = chains_[chain_id(v)];
    assert(chain_opt.has_value());
    return *chain_opt;
}

void ChainGroup::merge_chains_(Move const& v1, Move const& v2) {
    int id1 = chain_id(v1), id2 = chain_id(v2);
    if (! chain_tree_.unite(id1, id2)) return;

    int id_base = id1, id_new = id2;
    if (chain_tree_.root(id1) != id1)
        std::swap(id_base, id_new);

    Chain& base_chain = *chains_.at(id_base);
    Chain& new_chain = *chains_.at(id_new);

    base_chain.merge(new_chain);  // splice
    chain_list_.set_next(base_chain.tail, new_chain.head);
    base_chain.tail = new_chain.tail;
    assert(chain_tree_.root(id_new) == id_base);
    chains_[id_new].reset();
}

std::vector<Move> ChainGroup::chain_members(int id) const {
    auto p = chains_.at(id)->head;
    std::vector<Move> ret;
    ret.reserve(chain_size(id));
    ret.emplace_back(Move::from_raw(p, board_size_));
    while (auto next = chain_list_.next(p)) {
        p = *next;
        ret.emplace_back(Move::from_raw(p, board_size_));
    }
    return ret;
}

void ChainGroup::remove_chain_(Move const& vertex) {
    auto id = chain_id(vertex);
    auto stones(chain_members(id));

    for (auto const& v : stones) {
        for_each_4nbr(v, [&] (Move const& nbr) {
            if (stone_at(nbr) == Color::EMPTY) {
                return;
            }

            auto nbr_id = chain_id(nbr);
            if (nbr_id == id || !chains_[nbr_id]) { // myself stil alive or myself already removed
                return;
            }

            chain_at_(nbr).add_adjacent_empty(v);
        });
        remove_stone_(v);
        chain_tree_._reset_one(v());
        chain_list_.reset(v());
    }
    chains_[id].reset();
}

void ChainGroup::remove_stone_(Move const& v) {
    Color c = stone_at(v);

    set_stone_(Color::EMPTY, v);

    for_each_4nbr(v, [&] (Move const& nbr) {
        neighbor_counters_[nbr()].decrement(c);
    });

    hash_.update(c, v);
}

void ChainGroup::set_stone_(Color c, Move const& v) {
    stones_[v()] = c;
    if (c != Color::EMPTY) {
        if (c == Color::BLACK) black_stones_[v()] = 1;
        else white_stones_[v()] = 1;
        chain_tree_.enable(v());
        return;
    }

    black_stones_[v()] = white_stones_[v()] = 0;
}


std::string ChainGroup::to_string() const {
    std::stringstream ss;

    for (size_t i=0; i<chains_.size(); ++i) {
        const auto& maybe_chain = chains_[i];
        if (! maybe_chain)
            continue;
        auto const& chain = *maybe_chain;

        ss << chain << std::endl;
        for (auto const& v : chain_members(i)) {
            ss << Move::from_raw(v(), board_size_) << '(' << v() << ')' << ": " << chain_id(v) << ", ";
        }

        ss << std::endl;
    }

    return ss.str();
}

bool ChainGroup::check_internal_consistency() const {
    size_t bs = board_size_;
    auto size_ok = \
        stones_.size() == bs * bs
        && black_stones_.size() == bs * bs
        && white_stones_.size() == bs * bs
        && neighbor_counters_.size() == bs * bs
        && chains_.size() == bs * bs;
    if (not size_ok)
        return false;
    bool chain_ok = true;
    for_each_coordinate(board_size_, [&](Move const& v) {
        if (stone_at(v) == Color::EMPTY) return;
        auto& chain = chain_at(v);
        if (chain.liberty_count() == 0) chain_ok = false;
    });
    if (!chain_ok)
        return false;

    return true;
}

}  // namespace cygo
