#ifndef CYGO_MOVE_HPP
#define CYGO_MOVE_HPP

#include <string>
#include <cstddef>


namespace cygo {

class Move {
private:
    int raw_;
    int board_size_;

    explicit Move(int raw) noexcept;
    Move(int raw, int board_size);

public:
    Move() noexcept;

    static Move from_coordinate(int row, int column, int board_size);
    static Move from_gtp_string(std::string const& str, int board_size);
    static Move from_raw(int raw, int board_size);

    static Move ANY;
    static Move PASS;
    static Move INVALID;

    bool operator==(Move const& other) const;
    bool operator!=(Move const& other) const;
    int operator()() const;

    int row() const;
    int col() const;
    int board_size() const;
    int raw() const;

    Move n() const;
    Move s() const;
    Move w() const;
    Move e() const;

    bool is_on_edge() const;
    bool is_at_corner() const;

    std::string to_string() const;
};

inline bool operator<(Move const& lhs, Move const& rhs) {
    return lhs.raw() < rhs.raw();
}

inline std::ostream& operator<<(std::ostream& os, Move const& m) {
    os << m.to_string();
    return os;
}


template <typename F>
void for_each_4nbr(Move const& v, F f) {
    for (auto const& u : {v.n(), v.s(), v.w(), v.e()}) {
        if (u != Move::INVALID) {
            f(u);
        }
    }
}

template <typename F>
void for_each_coordinate(int board_size, F f) {
    int num_coordinates = board_size * board_size;

    for (int v = 0; v < num_coordinates; ++ v) {
        f(Move::from_raw(v, board_size));
    }
}

}


namespace std {
    template <>
    struct hash<cygo::Move> {
        std::size_t operator()(cygo::Move const& v) const {
            return std::hash<int>()(v.raw());
        }
    };

}

#endif //CYGO_MOVE_HPP
