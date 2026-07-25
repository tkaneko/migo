#ifndef CYGO_CHAIN_GROUP_HPP
#define CYGO_CHAIN_GROUP_HPP

#include "chain.hpp"
#include "color.hpp"
#include "neighbor_counter.hpp"
#include "move.hpp"
#include "zobrist_hash.hpp"

#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <optional>
#include <cstdint>
#include <cassert>

namespace cygo {

template <class T = uint16_t>
struct UFT {
    explicit UFT(T size) : tree(size) {
        for (int i = 0; i < size; ++i) tree[i] = std::make_pair(i, 0);
    }
    T root(T n) const {
        T parent = tree[n].first;
        return parent == n ? n : (tree[n].first = root(parent));
    }
    std::optional<T> root_or_empty(T n) const {
        T root_id = root(n);
        return empty(root_id) ? std::nullopt : std::make_optional(root_id);
    }

    T size(T n) const { return tree[n].second; }
    T empty(T n) const { return size(n) == 0; }
    void enable(T n) {
        if (size(n) > 0) throw std::invalid_argument("new on occupied node");
        tree[n].second = 1;
    }
    bool is_same_group(T a, T b) const { return root(a) == root(b); }
    /** merge two groups a and b, return new size if success (0 if already
     * united) */
    T unite(T a, T b) {
        assert(!empty(a) && !empty(b));
        T ra = root(a), rb = root(b);
        if (empty(ra) || empty(rb))
            throw std::invalid_argument("unite empty group");
        if (ra == rb) return 0;
        if (size(ra) < size(rb)) std::swap(ra, rb);
        // larger ra remains root
        tree[rb].first = ra;
        tree[ra].second += tree[rb].second;
        return tree[ra].second;
    }
    /** a sequence of calls must conform to the same condition as in reset_group
     */
    void _reset_one(T n) { tree[n] = std::make_pair(n, 0); }
    /** reset group of elements, range [first,last) should cover all members of
     * root(e) for e in the range */
    template <class Iter>
    void reset_group(Iter first, Iter last) {
        for (Iter p = first; first != last; ++p) {
            _reset_one(*p);
        }
    }

    mutable std::vector<std::pair<T, T>> tree;  // id, size
};

/** doubly linked list of uint16_t */
struct VertexList
{
    std::vector<uint16_t> edge; // forward, dst== edge[src]
    explicit VertexList(int size) : edge(size) {
        for (size_t i = 0; i < edge.size(); ++i)
            reset(i);
    }
    void reset(uint16_t id) {
        edge[id] = id; // nil
    }
    std::optional<int> next(int id) const {
        auto n = edge[id];
        return (n == id) ? std::nullopt : std::make_optional(n);
    }
    void set_next(uint16_t id, uint16_t next) { edge[id] = next; }
    void reset_next(uint16_t id) { set_next(id, id); }
};

class ChainGroup {
public:
    explicit ChainGroup(int board_size);

    void place_stone(Color color, Move const& vertex);

    Chain const& chain_at(Move const& v) const;
    Color stone_at(Move const& v) const;

    std::unordered_set<Move> const& empties() const;

    std::vector<uint8_t> const& black_board() const;
    std::vector<uint8_t> const& white_board() const;

    std::vector<Color> const& stones() const;

    int count_stones(Color color) const;
    int empty_count_around(Move const& v) const;

    bool has_chain_at(Move const& v) const;
    bool is_atari_group(Move const& v) const;
    bool is_eye_like(Color color, Move const& vertex) const;

    ZobristHash::ValueType hash() const;

    std::string to_string() const;

    bool check_internal_consistency() const;
    int chain_id(Move const& v) const {
        return chain_tree_.root(v());
    }
    auto chain_size(int id) const {
        if (chain_tree_.root(id) != id)
            std::invalid_argument("chain_size");
        return chain_tree_.size(id);
    };
    auto chain_size_at(Move const& v) const {
        return chain_size(chain_id(v));
    }
    std::optional<int> chain_id_or_empty(Move const& v) const {
        return chain_tree_.root_or_empty(v());
    }

    std::vector<Move> chain_members(int id) const;
    std::vector<Move> chain_members_at(Move const& v) const {
        return chain_members(chain_id(v));
    }

private:
    /** valid only for vertices with a stone */
    Chain& chain_at_(Move const& v);

    void set_stone_(Color c, Move const& v);

    void merge_chains_(Move const& v1, Move const& v2);
    void remove_chain_(Move const& v);
    void remove_stone_(Move const& v);

private:
    const int board_size_;

    ZobristHash hash_;

    std::vector<Color> stones_;
    std::vector<uint8_t> black_stones_;
    std::vector<uint8_t> white_stones_;

    UFT<> chain_tree_;
    /** vertex list for chains */
    VertexList chain_list_;
    std::vector<NeighborCounter> neighbor_counters_;

    std::vector<std::optional<Chain>> chains_; // vertex -> Chain | nullopt, defined only for the root of each group
};

}  // namespace  cygo

#endif //CYGO_CHAIN_GROUP_HPP
