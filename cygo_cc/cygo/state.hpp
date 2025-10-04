#ifndef CYGO_STATE_HPP
#define CYGO_STATE_HPP

#include "color.hpp"
#include "move.hpp"
#include "zobrist_hash.hpp"

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

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
    /** drop history keeping current stones
     * after this call, history(.).size() becomes (at most) 1, and
     * move_history(.) becomes empty()
     */
    void drop_history();

    std::unordered_set<Move> legal_moves(Color c, bool include_eyeish = false) const;

    bool is_legal(Move const& move, Color player) const;
    bool is_suicide_move(Move const& v, Color player) const;
    bool is_eye_like(Move v, Color player=Color::EMPTY) const;

    std::vector<Color> const& stones() const;

    std::vector<uint8_t> const& black_board() const;
    std::vector<uint8_t> const& white_board() const;
    std::list<std::vector<uint8_t>> const& history(Color c) const;

    std::size_t max_history_n() const;

    int board_size() const;
    bool superko_rule() const;

    double tromp_taylor_score(Color c, int8_t *board_zero_filled=nullptr) const;

    Move last_move() const;
    std::vector<Move> move_history(Color c) const;

    std::string to_string() const;

    ZobristHash::ValueType hash() const;

    std::string info() const;
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
