#include <stdexcept>

#include "neighbor_counter.hpp"

namespace cygo {

constexpr int black_shift = 0;
constexpr int white_shift = 4;
constexpr int empty_shift = 8;

constexpr unsigned int black_increment = 0xFF'FF'FF'01;
constexpr unsigned int white_increment = 0xFF'FF'FF'10;
constexpr unsigned int empty_increment = 0xFF'FF'FF'11;

constexpr unsigned int initial_bits = 0b0100u << empty_shift;

NeighborCounter::NeighborCounter() : bits_(initial_bits) { }

NeighborCounter::NeighborCounter(unsigned int bits) : bits_(bits) { }

NeighborCounter NeighborCounter::create(unsigned int black, unsigned int white, unsigned int empty) {
    return NeighborCounter(
            (black << black_shift) + (white << white_shift) + (empty << empty_shift)
    );
}

NeighborCounter NeighborCounter::empty() {
    return NeighborCounter(initial_bits);
}

NeighborCounter NeighborCounter::empty_on_edge() {
    return NeighborCounter(initial_bits + empty_increment);
}

NeighborCounter NeighborCounter::empty_at_corner() {
    return NeighborCounter(initial_bits + empty_increment + empty_increment);
}

void NeighborCounter::increment(Color c) {
    if (c == Color::BLACK) {
        bits_ += black_increment;
        return;
    }

    if (c == Color::WHITE) {
        bits_ += white_increment;
        return;
    }
}

void NeighborCounter::decrement(Color c) {
    if (c == Color::BLACK) {
        bits_ -= black_increment;
        return;
    }

    if (c == Color::WHITE) {
        bits_ -= white_increment;
        return;
    }
}

bool NeighborCounter::is_eye_like(Color c) const {
    if (c == Color::BLACK) {
        return (bits_ & 0b0000'0100) != 0x0;
    }

    if (c == Color::WHITE) {
        return (bits_ & 0b0100'0000) != 0x0;
    }

    return false;
}

int NeighborCounter::empty_count() const {
    return static_cast<int>( (bits_ >> empty_shift) & 0b0111 );
}

int NeighborCounter::player_count(Color c) const {
    constexpr unsigned int mask = 0b0111;

    if (c == Color::BLACK) {
        return static_cast<int>( (bits_ >> black_shift) & mask );
    }

    if (c == Color::WHITE) {
        return static_cast<int>( (bits_ >> white_shift) & mask );
    }

    throw std::invalid_argument("");
}


}  // namespace cygo