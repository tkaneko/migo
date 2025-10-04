#ifndef CYGO_CHAIN_HPP
#define CYGO_CHAIN_HPP

#include "move.hpp"

#include <list>

namespace cygo {

class Chain {
public:
    static Chain NIL_CHAIN;

    Chain();

    explicit Chain(Move const& v);

    void add_adjacent_opponent(Move const& v);
    void add_adjacent_ally(Move const& v);
    void add_adjacent_empty(Move const& v);

    void merge(Chain& other);

    bool is_captured() const;
    bool is_in_atari() const;

    Move atari_vertex() const;

    std::list<Move> const& members() const;

    int liberty_count() const;
    int size() const;

    bool equals(const Chain& other);  // slow, intended only for tests

    int hash() const;

    std::string to_string() const;

private:
    void add_liberty(Move const& v);
    void subtract_liberty(Move const &v);

private:
    std::list<Move> members_;

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
