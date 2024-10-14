#include "state_impl.hpp"

#include <algorithm>
#include <queue>
#include <vector>


namespace cygo {

StateImpl::StateImpl(int board_size, bool superko_rule) :
        board_size_(board_size),
        superko_rule_(superko_rule),
        chain_group_(board_size),
        ko_vertex_(Move::ANY)
{ }


void StateImpl::make_move(Color c, Move const &v) {
    ko_vertex_ = Move::ANY;
    last_plays_[c] = v;

    if (v == Move::PASS) {
        color_move_history_[c].push_back(v);
        move_history_.push_back(v);
        return;
    }

    bool played_in_opponent_eye_like = chain_group_.is_eye_like(opposite_color(c), v);

    if (played_in_opponent_eye_like) {
        for_each_4nbr(v, [&] (Move const& nbr) {
            auto const& chain = chain_group().chain_at(nbr);

            if (chain.is_in_atari() and chain.size() == 1) {
                ko_vertex_ = nbr;
            }
        });
    }

    chain_group_.place_stone(c, v);

    hash_history_.emplace(chain_group_.hash());
    color_move_history_[c].push_back(v);
    move_history_.push_back(v);
}

std::unordered_set<Move> StateImpl::legal_moves(Color c, bool include_eye_likes) const {
    std::unordered_set<Move> legals;

    for (auto const& v: chain_group_.empties()) {
        if (is_legal(c, v) and (not is_eye_like(c, v) or include_eye_likes)) {
            legals.emplace(v);
        }
    }

    return legals;
}

bool StateImpl::is_eye_like(Color c, Move const &v) const {
    return chain_group_.is_eye_like(c, v);
}

ChainGroup const& StateImpl::chain_group() const {
    return chain_group_;
}

std::vector<Move> StateImpl::move_history(Color c) const {
    if (c == Color::EMPTY)
        return move_history_;
    
    if (color_move_history_.count(c) == 0) {
        return std::vector<Move>();
    }

    return color_move_history_.at(c);
}

bool StateImpl::is_legal(Color c, Move const& v) const {
    if (v == Move::PASS) {
        return true;
    }
    if (chain_group_.stone_at(v) != Color::EMPTY) {
        return false;
    }
    if (v == ko_vertex_) {
        return false;
    }
    if (is_suicide_move(c, v)) {
        return false;
    }

    return not superko_rule_ or not is_positional_superko(c, v);
}

bool StateImpl::is_suicide_move(Color c, Move const &v) const {
    if (chain_group_.empty_count_around(v) > 0) {
        return false;
    }

    bool suicide_move = true;

    for_each_4nbr(v, [&] (Move const& nbr) {
        bool nbr_is_atari = chain_group_.is_atari_group(nbr);
        bool nbr_is_ally = chain_group_.stone_at(nbr) == c;

        suicide_move &= (nbr_is_atari == nbr_is_ally);
    });

    return suicide_move;
}

bool StateImpl::is_positional_superko(Color c, Move const& v) const {
    if (color_move_history_.count(c) == 0) {
        return false;
    }

    auto const& move_history = color_move_history_.at(c);

    if (std::find(std::begin(move_history), std::end(move_history), v)
        == std::end(move_history)) {
        return false;
    }

    StateImpl copied(*this);
    copied.superko_rule_ = false;
    copied.make_move(c, v);

    return hash_history_.count(copied.hash()) != 0;
}

int StateImpl::board_size() const {
    return board_size_;
}

bool StateImpl::superko_rule() const {
    return superko_rule_;
}

double StateImpl::tromp_taylor_score(Color color) const {
    std::map<Color, int> score = {
            {Color::WHITE, 0},
            {Color::BLACK, 0}
    };

    for (Color c : {color, opposite_color(color)}) {
        std::queue<Move> queue;
        std::unordered_set<Move> visited;

        for_each_coordinate(board_size_, [&] (Move const& v) {
            if (chain_group_.stone_at(v) == c) {
                queue.push(v);
                visited.emplace(v);
            }
        });

        while (not queue.empty()) {
            Move v(queue.front()); queue.pop();
            score[c] += 1;

            for_each_4nbr(v, [&] (Move const& nbr) {
                if (visited.count(nbr) == 0 && chain_group_.stone_at(nbr) == Color::EMPTY) {
                    queue.push(nbr);
                    visited.emplace(nbr);
                }
            });
        }
    }

    return score[color] - score[opposite_color(color)];
}

ZobristHash::ValueType StateImpl::hash() const {
    return chain_group_.hash();
}

}
