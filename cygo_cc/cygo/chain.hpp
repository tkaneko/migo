#ifndef CYGO_CHAIN_HPP
#define CYGO_CHAIN_HPP

#include <unordered_set>

#include "move.hpp"


namespace cygo {

class Chain {
public:
    static Chain NIL_CHAIN;

    Chain();

    explicit Chain(Move const& v);

    void add_adjacent_opponent(Move const& v);
    void add_adjacent_ally(Move const& v);
    void add_adjacent_empty(Move const& v);

    void merge(Chain const& other);

    bool is_captured() const;
    bool is_in_atari() const;

    Move atari_vertex() const;

    std::unordered_set<Move> const& members() const;
    std::unordered_set<Move> const& adjacent_empties() const;
    std::unordered_set<Move> const& adjacent_opponents() const;

    int liberty_count() const;
    int size() const;

    bool operator==(Chain const& other);
    bool operator!=(Chain const& other);

    int hash() const;

    std::string to_string() const;

private:
    void add_liberty(Move const& v);
    void subtract_liberty(Move const &v);

private:
    std::unordered_set<Move> members_;
    std::unordered_set<Move> adjacent_empties_;
    std::unordered_set<Move> adjacent_opponents_;

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
