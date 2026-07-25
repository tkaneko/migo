#ifndef CYGO_CHAIN_HPP
#define CYGO_CHAIN_HPP

#include "move.hpp"

namespace cygo {

class Chain {
public:
    static Chain NIL_CHAIN;
    int16_t head=-1, tail=-1;       // for coordination with VertexList

    Chain();

    Chain(Move const& v, int id);

    void add_adjacent_opponent(Move const& v);
    void add_adjacent_ally(Move const& v);
    void add_adjacent_empty(Move const& v);

    void merge(Chain& other);

    bool is_captured() const;
    bool is_in_atari() const;

    Move atari_vertex(int board_size) const;

    int liberty_count() const;

    int hash() const;

    std::string to_string() const;
    size_t size2() const {
        if (head < 0) return 0;
        if (head == tail) return 1;
        return 2;  // or more
    }

private:
    void add_liberty(Move const& v);
    void subtract_liberty(Move const &v);

private:
    int liberty_count_;
    int liberty_sum_;
    int liberty_sum_sq_;
};

inline std::ostream& operator<<(std::ostream& os, Chain const& chain) {
    os << chain.to_string();
    return os;
}

}  // namespace cygo


namespace std {
    template <>
    struct hash<cygo::Chain> {
        std::size_t operator()(cygo::Chain const& key) const {
            return hash<int>()(key.hash());
        }
    };
}


#endif //CYGO_CHAIN_HPP
