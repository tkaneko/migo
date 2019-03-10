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
class State;

class IllegalMoveException : public std::invalid_argument {
public:
    explicit IllegalMoveException(std::string const &s);
};

struct History {
    const int history_length;

    std::list<std::vector<int>> black_history;
    std::list<std::vector<int>> white_history;

    explicit History(int history_length);

    void add(State const& state);

    std::list<std::vector<int>> const& history(Color c) const;
};

class State {
public:
    explicit State(int board_size, double komi, bool superko_rule, bool retain_history);
    explicit State(int board_size, double komi, bool superko_rule, int history_length);

    State(State const&);

    State& operator=(State const&) = delete;

    void make_move(Move const &move, Color player = Color::EMPTY);

    std::unordered_set<Move> legal_moves(Color c, bool include_eyeish = false) const;

    bool is_legal(Move const& move, Color player) const;

    std::vector<Color> const& stones() const;

    std::vector<int> const& black_board() const;
    std::vector<int> const& white_board() const;

    History const& history() const;

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
    History history_;
    Move last_move_;

    std::shared_ptr<StateImpl> state_;
};

}  // namespace cygo

#endif //CYGO_STATE_HPP
