#include <memory>
#include <sstream>

#include "state.hpp"
#include "state_impl.hpp"


namespace cygo {


IllegalMoveException::IllegalMoveException(std::string const& s) : invalid_argument(s) {}


History::History(int history_length) : history_length(history_length) { }

void History::add(State const& state) {
    if (history_length == 0) {
        return;
    }

    if (black_history.size() == history_length) {
        black_history.pop_back();
        white_history.pop_back();
    }

    black_history.push_front(state.black_board());
    white_history.push_front(state.white_board());
}

std::list<std::vector<int>> const& History::history(Color c) const {
    return (c == Color::BLACK) ? black_history : white_history;
}


State::State(int board_size, double komi, bool superko_rule, int history_length) :
        komi(komi),
        current_player(Color::BLACK),
        history_(history_length),
        last_move_(Move::INVALID),
        state_(new StateImpl(board_size, superko_rule))
{ }

State::State(int board_size, double komi, bool superko_rule, bool retain_history) :
        State(board_size, komi, superko_rule, (retain_history) ? 10 : 0)
{ }

State::State(State const& other) :
        komi(other.komi),
        current_player(other.current_player),
        history_(other.history_),
        last_move_(other.last_move_)
{
    state_ = std::make_shared<StateImpl>(*other.state_);
}

void State::make_move(Move const& move, Color player) {
    if (player == Color::EMPTY) {
        player = current_player;
    }

    if (not state_->is_legal(player, move)) {
        throw IllegalMoveException("Illegal move");
    }

    state_->make_move(player, move);
    last_move_ = move;

    history_.add(*this);

    current_player = opposite_color(current_player);
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

std::vector<Color> const& State::stones() const {
    return state_->chain_group().stones();
}

std::vector<int> const& State::black_board() const {
    return state_->chain_group().black_board();
}

std::vector<int> const& State::white_board() const {
    return state_->chain_group().white_board();
}

History const& State::history() const {
    return history_;
}

int State::board_size() const {
    return state_->board_size();
}

bool State::superko_rule() const {
    return state_->superko_rule();
}

double State::tromp_taylor_score(Color c) const {
    if (c == Color::EMPTY) {
        c = current_player;
    }

    double score = state_->tromp_taylor_score(c);

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
