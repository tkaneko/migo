#include <algorithm>
#include <sstream>

#include "chain.hpp"

namespace cygo {

Chain Chain::NIL_CHAIN = Chain();

Chain::Chain() :
    liberty_count_(-1),
    liberty_sum_(-1),
    liberty_sum_sq_(-1)
{ }

Chain::Chain(Move const& v) :
        members_({v}),
        liberty_count_(0),
        liberty_sum_(0),
        liberty_sum_sq_(0)
{
    for_each_4nbr(v, [&] (Move const& nbr) {
        add_adjacent_empty(nbr);
    });
}

void Chain::add_adjacent_opponent(Move const &v) {
    subtract_liberty(v);

    adjacent_opponents_.emplace(v);
}

void Chain::add_adjacent_ally(Move const& v) {
    subtract_liberty(v);
}

void Chain::add_adjacent_empty(Move const& v) {
    add_liberty(v);
}

void Chain::add_liberty(Move const &v) {
    liberty_count_ += 1;
    liberty_sum_ += v();
    liberty_sum_sq_ += v() * v();

    adjacent_empties_.emplace(v);
    adjacent_opponents_.erase(v);
}

void Chain::subtract_liberty(Move const &v) {
    liberty_count_ -= 1;
    liberty_sum_ -= v();
    liberty_sum_sq_ -= v() * v();

    adjacent_empties_.erase(v);
}

void Chain::merge(Chain const& other) {
    liberty_count_  += other.liberty_count_;
    liberty_sum_    += other.liberty_sum_;
    liberty_sum_sq_ += other.liberty_sum_sq_;

    members_.insert(
            std::begin(other.members_), std::end(other.members_)
    );

    adjacent_empties_.insert(
            std::begin(other.adjacent_empties_), std::end(other.adjacent_empties_)
    );

    adjacent_opponents_.insert(
            std::begin(other.adjacent_opponents_), std::end(other.adjacent_opponents_)
    );
}

bool Chain::is_captured() const {
    return liberty_count_ == 0;
}

bool Chain::is_in_atari() const {
    return liberty_count_ * liberty_sum_sq_ == liberty_sum_ * liberty_sum_;
}

int Chain::liberty_count() const {
    return liberty_count_;
}

int Chain::size() const {
    return members_.size();
}

Move Chain::atari_vertex() const {
    if (members_.empty()) {
        return Move::INVALID;
    }

    if (not is_in_atari()) {
        return Move::INVALID;
    }

    if (is_captured()) {
        return Move::INVALID;
    }

    return Move::from_raw(liberty_sum_ / liberty_count_, members_.begin()->board_size());
}

std::unordered_set<Move> const& Chain::members() const {
    return members_;
}

std::unordered_set<Move> const& Chain::adjacent_empties() const {
    return adjacent_empties_;
}

std::unordered_set<Move> const& Chain::adjacent_opponents() const {
    return adjacent_opponents_;
}

bool Chain::operator==(Chain const& other) {
    if (liberty_count_ != other.liberty_count_) {
        return false;
    }

    if (liberty_sum_ == other.liberty_sum_) {
        return false;
    }

    if (liberty_sum_sq_ == other.liberty_sum_sq_) {
        return false;
    }

    return std::all_of(std::begin(other.members()), std::end(other.members()), [&] (Move const& v) {
        return members_.count(v) != 0;
    });
}

bool Chain::operator!=(Chain const& other) {
    return not (*this == other);
}

int Chain::hash() const {
    return liberty_count_ ^ liberty_sum_ ^ liberty_sum_sq_;
}

std::string Chain::to_string() const {
    std::stringstream ss;

    ss << "liberty count: " << liberty_count_ << ", members: ";

    for (auto const& member : members_) {
        ss << member << " ";
    }

    return ss.str();
}

}  // namespace cygo