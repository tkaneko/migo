#include <cassert>
#include <numeric>
#include <set>
#include <sstream>

#include "chain_group.hpp"


namespace cygo {

ChainGroup::ChainGroup(int board_size) :
        board_size_(board_size),
        stones_(board_size * board_size, Color::EMPTY),
        black_stones_(board_size * board_size, 0),
        white_stones_(board_size * board_size, 0),
        chain_ids_(board_size * board_size),
        neighbor_counters_(board_size * board_size),
        chains_(board_size * board_size)
{
    for_each_coordinate(board_size_, [&] (Move const& v) {
        empties_.emplace(v);
        chain_ids_[v()] = v();

        if (v.is_at_corner()) {
            neighbor_counters_[v()] = NeighborCounter::empty_at_corner();
        }
        else if (v.is_on_edge()) {
            neighbor_counters_[v()] = NeighborCounter::empty_on_edge();
        }
        else {
            neighbor_counters_[v()] = NeighborCounter::empty();
        }
    });
}

void ChainGroup::place_stone(Color c, Move const& v) {
    hash_.update(c, v);

    set_stone(c, v);

    auto id = chain_ids_[v()];
    chains_[id].emplace(v);

    for_each_4nbr(v, [&] (Move const& nbr) {
        neighbor_counters_[nbr()].increment(c);
    });

    // update liberty counts first
    for_each_4nbr(v, [&] (Move const& nbr) {
        Color nbr_color = stone_at(nbr);

        if (nbr_color == Color::EMPTY) {
            return;
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
            return;
        }

        if (nbr_color == c) {
            merge_chains(v, nbr);
        }
        else if (chain_at(nbr).is_captured()){
            remove_chain(nbr);
        }
    });
}

Chain const& ChainGroup::chain_at(Move const& v) const {
    return *chains_.at(chain_ids_.at(v()));
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
    return chains_.at(chain_ids_.at(v())).has_value();
}

ZobristHash::ValueType ChainGroup::hash() const {
    return hash_.hash_value();
}

std::unordered_set<Move> const& ChainGroup::empties() const {
    return empties_;
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
    assert(chains_.at(chain_ids_[v()]));

    return *chains_[chain_ids_[v()]];
}

void ChainGroup::merge_chains(Move const& v1, Move const& v2) {
    if (chain_ids_[v1()] == chain_ids_[v2()]) {
        return;
    }

    Move v_base, v_new;

    if (chain_at(v1).size() > chain_at(v2).size()) {
        v_base = v1; v_new = v2;
    }
    else {
        v_base = v2; v_new = v1;
    }

    Chain& base_chain = chain_at_(v_base);
    Chain& new_chain = chain_at_(v_new);

    auto new_members = new_chain.members().begin();

    base_chain.merge(new_chain); // splice

    for (auto p=new_members; p!=base_chain.members().end(); ++p) {
        auto v = *p;
        chains_[chain_ids_[v()]].reset();

        chain_ids_[v()] = chain_ids_[v_base()];
    }
}

void ChainGroup::remove_chain(Move const &vertex) {
    Chain& chain = chain_at_(vertex);

    auto stones(chain.members());

    auto id = chain_ids_[vertex()];

    for (auto const& v : stones) {
        for_each_4nbr(v, [&] (Move const& nbr) {
            if (stone_at(nbr) == Color::EMPTY) {
                return;
            }

            if (&chain_at_(nbr) == &chain) {
                return;
            }

            chain_at_(nbr).add_adjacent_empty(v);
        });

        chain_ids_[v()] = v();

        remove_stone(v);
    }

    chains_[id].reset();

}

void ChainGroup::remove_stone(Move const& v) {
    Color c = stone_at(v);

    set_stone(Color::EMPTY, v);

    for_each_4nbr(v, [&] (Move const& nbr) {
        neighbor_counters_[nbr()].decrement(c);
    });

    hash_.update(c, v);
}

void ChainGroup::set_stone(Color c, Move const& v) {
    if (c == Color::BLACK) {
        stones_[v()] = Color::BLACK;
        black_stones_[v()] = 1;
        empties_.erase(v);
        return;
    }
    if (c == Color::WHITE) {
        stones_[v()] = Color::WHITE;
        white_stones_[v()] = 1;
        empties_.erase(v);
        return;
    }

    stones_[v()] = Color::EMPTY;
    black_stones_[v()] = white_stones_[v()] = 0;
    empties_.emplace(v);
}

std::string ChainGroup::to_string() const {
    std::stringstream ss;

    for (auto const& maybe_chain : chains_) {
        if (! maybe_chain)
            continue;
        auto const& chain = *maybe_chain;

        ss << chain << std::endl;
        for (auto const& v : chain.members()) {
            ss << v() << ": " << chain_ids_[v()] << ", ";
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
        && chain_ids_.size() == bs * bs
        && neighbor_counters_.size() == bs * bs
        && chains_.size() == bs * bs;
    if (not size_ok)
        return false;

    return true;
}

}  // namespace cygo
