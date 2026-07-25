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

Chain::Chain(Move const& v, int id)
    : head(id),
      tail(id),
      liberty_count_(0),
      liberty_sum_(0),
      liberty_sum_sq_(0) {
    for_each_4nbr(v, [&](Move const& nbr) { add_adjacent_empty(nbr); });
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
}

bool Chain::is_captured() const {
    return liberty_count_ == 0;
}

bool Chain::is_in_atari() const {
    // atari <==>
    // - liberty_count_ == 1,
    // - liberty_sum_sq_ == liberty_sum_ * liberty_sum_
    // or
    // - liberty_count_ == 2, but the same vertex is the unique liberty
    // - liberty_sum_sq_ * 2 == 2(v^2+v^2) == 4v^2 == liberty_sum_ * liberty_sum_
    // or works for liberty_count_ == 3 or 4 for the same vertex is the unique liberty
    return liberty_count_ * liberty_sum_sq_ == liberty_sum_ * liberty_sum_;
}

int Chain::liberty_count() const {
    return liberty_count_;
}

Move Chain::atari_vertex(int board_size) const {
    if (liberty_count_ < 0) {
        return Move::INVALID;
    }

    if (not is_in_atari()) {
        return Move::INVALID;
    }

    if (is_captured()) {
        return Move::INVALID;
    }

    return Move::from_raw(liberty_sum_ / liberty_count_, board_size);
}

int Chain::hash() const {
    return liberty_count_ ^ liberty_sum_ ^ liberty_sum_sq_;
}

std::string Chain::to_string() const {
    std::stringstream ss;

    ss << "liberty count: " << liberty_count_ << ", members: ";
    return ss.str();
}

}  // namespace cygo
