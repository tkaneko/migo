#ifndef CYGO_ZOBRIST_HASH_HPP
#define CYGO_ZOBRIST_HASH_HPP

#include <random>

#include "color.hpp"
#include "move.hpp"


namespace cygo {

class ZobristHash {
public:
    using ValueType = std::mt19937_64::result_type;
    static constexpr std::size_t MAX_SIZE = 25 * 25;

private:
    ValueType hash_value_;

public:
    explicit ZobristHash(ValueType initial_value=0u);

    void update(Color c, Move const& v);

    ValueType hash_value() const;

    static ValueType calculate_hash(std::vector<Move> const& blacks, std::vector<Move> const& whites);
};

}  // namespace cygo

#endif //CYGO_ZOBRIST_HASH_HPP
