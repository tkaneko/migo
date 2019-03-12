#ifndef CYGO_STATE_HPP
#define CYGO_STATE_HPP

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

#include "color.hpp"
#include "move.hpp"
#include "zobrist_hash.hpp"


namespace cygo {

class StateImpl;

class IllegalMoveException : public std::invalid_argument {
public:
    explicit IllegalMoveException(std::string const &s);
};


class State {
public:
    explicit State(int board_size, double komi, bool superko_rule, std::size_t max_history_n);

    State(State const&);

    State& operator=(State const&) = delete;

    ~State();

    void make_move(Move const &move, Color player = Color::EMPTY);

    std::unordered_set<Move> legal_moves(Color c, bool include_eyeish = false) const;

    bool is_legal(Move const& move, Color player) const;

    std::vector<Color> const& stones() const;

    std::vector<int> const& black_board() const;
    std::vector<int> const& white_board() const;
    std::list<std::vector<int>> const& history(Color c) const;

    std::size_t max_history_n() const;

    int board_size() const;
    bool superko_rule() const;

    double tromp_taylor_score(Color c) const;

    Move last_move() const;
    std::vector<Move> move_history(Color c) const;

    std::string to_string() const;

    ZobristHash::ValueType hash() const;

public:
    double komi;
    Color current_player;

private:
    class History;

    Move last_move_;

    std::unique_ptr<History> history_;
    std::unique_ptr<StateImpl> state_;
};

}  // namespace cygo

#endif //CYGO_STATE_HPP
