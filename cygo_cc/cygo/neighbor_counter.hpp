#ifndef CYGO_NEIGHBOR_COUNTER_HPP
#define CYGO_NEIGHBOR_COUNTER_HPP

#include "color.hpp"
#include <cstdint>

namespace cygo {

typedef uint16_t neighbor_bits_t;

class NeighborCounter {
public:
    NeighborCounter();

    void increment(Color c);
    void decrement(Color c);

    int empty_count() const;
    int player_count(Color c) const;

    bool is_eye_like(Color c) const;

    static NeighborCounter empty();
    static NeighborCounter empty_on_edge();
    static NeighborCounter empty_at_corner();

private:
    neighbor_bits_t bits_;

    explicit NeighborCounter(neighbor_bits_t b);

    static NeighborCounter create(neighbor_bits_t black, neighbor_bits_t white, neighbor_bits_t empty);
};

}

#endif //CYGO_NEIGHBOR_COUNTER_HPP
