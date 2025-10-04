#include <cassert>
#include <memory>
#include <sstream>

#include "state.hpp"
#include "state_impl.hpp"


namespace cygo {


IllegalMoveException::IllegalMoveException(std::string const& s) : invalid_argument(s) {}


class State::History {
private:
    std::list<std::vector<uint8_t>> black_history_;
    std::list<std::vector<uint8_t>> white_history_;

public:
    const std::size_t max_buffer_len;

    explicit History(std::size_t max_buffer_len) : max_buffer_len(max_buffer_len) { }

    void on_after_make_move(State const& state) {
        black_history_.push_front(state.black_board());
        white_history_.push_front(state.white_board());

        while (black_history_.size() > max_buffer_len) {
            black_history_.pop_back();
        }

        while (white_history_.size() > max_buffer_len) {
            white_history_.pop_back();
        }

        assert(black_history_.size() == white_history_.size());
    }

    void drop_history() {
        if (black_history_.size() > 1)
            black_history_.resize(1);
        if (white_history_.size() > 1)
            white_history_.resize(1);
    }

    std::list<std::vector<uint8_t>> const& get(Color c) const {
        if (c == Color::BLACK) {
            return black_history_;
        }

        if (c == Color::WHITE) {
            return white_history_;
        }

        throw std::invalid_argument("");
    }
};


State::State(int board_size, double komi, bool superko_rule, std::size_t max_history_n) :
        komi(komi),
        current_player(Color::BLACK),
        last_move_(Move::INVALID),
        history_(std::make_unique<History>(max_history_n + 1)),  // +1 for the current state
        state_(std::make_unique<StateImpl>(board_size, superko_rule))
{ }

State::State(State const& other) :
        komi(other.komi),
        current_player(other.current_player),
        last_move_(other.last_move_),
        history_(std::make_unique<History>(*other.history_)),
        state_(std::make_unique<StateImpl>(*other.state_))
{ }

State::~State() = default;

void State::make_move(Move const& move, Color player) {
    if (player == Color::EMPTY) {
        player = current_player;
    }

    std::string err;
    if (not state_->is_legal(player, move, &err)) {
        throw IllegalMoveException("Illegal move " + err);
    }

    state_->make_move(player, move);
    last_move_ = move;

    history_->on_after_make_move(*this);

    current_player = opposite_color(current_player);
}

void State::drop_history() {
    history_->drop_history();
    state_->drop_history();
    last_move_ = Move::INVALID;
}  

std::unordered_set<Move> State::legal_moves(Color c, bool include_eyes) const {
    if (c == Color::EMPTY) {
        c = current_player;
    }

    return state_->legal_moves(c, include_eyes);
}

bool State::is_legal(Move const& move, Color player) const {
    if (player == Color::EMPTY) {
        player = current_player;
    }

    return state_->is_legal(player, move);
}

bool State::is_suicide_move(Move const& v, Color c) const {
    if (c == Color::EMPTY) {
        c = current_player;
    }
    return state_->is_suicide_move(c, v);
}

bool State::is_eye_like(Move v, Color c) const {
    if (c == Color::EMPTY)
        c = current_player;
    return state_->is_eye_like(c, v)
        && state_->is_suicide_move(opposite_color(c), v);
}

std::vector<Color> const& State::stones() const {
    return state_->chain_group().stones();
}

std::vector<uint8_t> const& State::black_board() const {
    return state_->chain_group().black_board();
}

std::vector<uint8_t> const& State::white_board() const {
    return state_->chain_group().white_board();
}

std::list<std::vector<uint8_t>> const& State::history(Color c) const {
    return history_->get(c);
}

std::size_t State::max_history_n() const {
    return history_->max_buffer_len - 1;
}

int State::board_size() const {
    return state_->board_size();
}

bool State::superko_rule() const {
    return state_->superko_rule();
}

double State::tromp_taylor_score(Color c, int8_t *board_zero_filled) const {
    if (c == Color::EMPTY) {
        c = current_player;
    }

    double score = state_->tromp_taylor_score(c, board_zero_filled);

    return (c == Color::BLACK) ? score - komi : score + komi;
}

Move State::last_move() const {
    return last_move_;
}

std::vector<Move> State::move_history(Color c) const {
    return state_->move_history(c);
}

std::string State::to_string() const {
    static const std::string COLUMNS("ABCDEFGHJKLMNOPQRSTUVWXYZ");

    std::ostringstream os;

    os << ((board_size() < 10) ? " " : "  ");

    for (int col = 0; col < board_size(); ++col) {
        os << " " << COLUMNS[col] << " ";
    }

    os << std::endl;

    auto stones = this->stones();

    for (int row = board_size() - 1; row >= 0; --row) {
        if (board_size() >= 10 and (row + 1) < 10) { os << " "; }
        os << (row + 1);

        for (int col = 0; col < board_size(); ++col) {
            Move m = Move::from_coordinate(row, col, board_size());
            Color c = stones.at(m());

            char stone;
            if (c == Color::WHITE) {
                stone = 'O';
            }
            else if (c == Color::BLACK) {
                stone = 'X';
            }
            else {
                stone = '.';
            }

            if (m == last_move_) {
                os << "(" << stone << ")";
            }
            else {
                os << " " << stone << " ";
            }

        }

        os << (row + 1);
        os << std::endl;
    }

    os << ((board_size() < 10) ? " " : "  ");

    for (int col = 0; col < board_size(); ++col) {
        os << " " << COLUMNS[col] << " ";
    }

    os << std::endl;

    return os.str();
}

ZobristHash::ValueType State::hash() const {
    return state_->hash();
}

}

std::string cygo::State::info() const {
  return state_->info();
}

