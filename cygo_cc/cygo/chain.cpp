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
}

void Chain::subtract_liberty(Move const &v) {
    liberty_count_ -= 1;
    liberty_sum_ -= v();
    liberty_sum_sq_ -= v() * v();
}

void Chain::merge(Chain& other) {
    liberty_count_  += other.liberty_count_;
    liberty_sum_    += other.liberty_sum_;
    liberty_sum_sq_ += other.liberty_sum_sq_;

    members_.splice(members_.end(), other.members_);
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

std::list<Move> const& Chain::members() const {
    return members_;
}

bool Chain::equals(const Chain& other) {
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
        return std::find(members_.begin(), members_.end(), v) != members_.end();
    });
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
