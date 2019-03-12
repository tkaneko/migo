#include <array>
#include <random>

#include "zobrist_hash.hpp"


namespace cygo {

constexpr std::size_t MAX_SIZE = 25 * 25;

std::array<std::uint64_t, MAX_SIZE> generate_hash_table(std::uint_fast64_t seed) {
    std::mt19937_64 engine(seed);
    std::array<std::uint64_t, MAX_SIZE> ret;

    for (auto& v : ret) {
        v = engine();
    }

    return ret;
};

static std::array<std::uint_fast64_t, MAX_SIZE> black_table = generate_hash_table(0);
static std::array<std::uint_fast64_t, MAX_SIZE> white_table = generate_hash_table(1);

ZobristHash::ZobristHash(std::uint_fast64_t initial_value) : hash_value_(initial_value) { }

void ZobristHash::update(Color c, Move const &v) {
    if (c == Color::BLACK) {
        hash_value_ ^= black_table.at(v.raw());
        return;
    }

    if (c == Color::WHITE) {
        hash_value_ ^= white_table.at(v.raw());
        return;
    }
}

std::uint_fast64_t ZobristHash::hash_value() const {
    return hash_value_;
}

}  // namespace cygo