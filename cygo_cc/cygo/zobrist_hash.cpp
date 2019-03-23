#include <array>
#include <random>

#include "zobrist_hash.hpp"


namespace cygo {

std::array<ZobristHash::ValueType, ZobristHash::MAX_SIZE> generate_hash_table(std::uint_fast64_t seed) {
    std::mt19937_64 engine(seed);
    std::array<ZobristHash::ValueType, ZobristHash::MAX_SIZE> ret = {};

    for (auto& v : ret) {
        v = engine();
    }

    return ret;
};

static std::array<ZobristHash::ValueType, ZobristHash::MAX_SIZE> black_table = generate_hash_table(0);
static std::array<ZobristHash::ValueType, ZobristHash::MAX_SIZE> white_table = generate_hash_table(1);

ZobristHash::ZobristHash(ZobristHash::ValueType initial_value) : hash_value_(initial_value) { }

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

ZobristHash::ValueType ZobristHash::hash_value() const {
    return hash_value_;
}

ZobristHash::ValueType ZobristHash::calculate_hash(std::vector<cygo::Move> const& blacks,
                                                   std::vector<cygo::Move> const& whites) {
    ZobristHash::ValueType hash = 0;

    for (auto const& m : blacks) {
        hash ^= black_table.at(m.raw());
    }

    for (auto const& m : whites) {
        hash ^= white_table.at(m.raw());
    }

    return hash;
}

}  // namespace cygo