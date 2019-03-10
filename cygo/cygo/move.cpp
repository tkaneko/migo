#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

#include "move.hpp"


namespace cygo {

static const std::string COLUMNS("ABCDEFGHJKLMNOPQRSTUVWXYZ");

Move Move::ANY = Move(-1);
Move Move::PASS = Move(-2);
Move Move::INVALID = Move();

Move::Move(int raw) noexcept : raw_(raw), board_size_(0) { }

Move::Move(int raw, int board_size) : raw_(raw), board_size_(board_size) { }

Move::Move() noexcept : raw_(-3) { }

Move Move::from_coordinate(int row, int col, int board_size) {
    if (board_size <= 0) {
        throw std::invalid_argument("board_size should be greater than 0");
    }

    if (not (0 <= row and row < board_size)) {
        throw std::invalid_argument(std::string("Given row is out of bound: ") + std::to_string(row));
    }

    if (not (0 <= col and col < board_size)) {
        throw std::invalid_argument(std::string("Given column is out of bound: ") + std::to_string(col));
    }

    return Move(row * board_size + col, board_size);
}

Move Move::from_gtp_string(std::string const &str, int board_size) {
    std::string s(str);
    std::transform(std::begin(s), std::end(s), std::begin(s), ::toupper);

    if (s == "PASS") {
        return Move::PASS;
    }

    std::istringstream parser(s);

    char col;
    int row;

    if (!(parser >> col >> row)) {
        throw std::invalid_argument("Invalid GTP string");
    }

    return Move::from_coordinate(row - 1, COLUMNS.find(col), board_size);
}

Move Move::from_raw(int raw, int board_size) {
    if (board_size <= 0) {
        throw std::invalid_argument(std::string("Board size should be greater than 0: ") + std::to_string(board_size));
    }

    if (not (0 <= raw and raw < board_size * board_size)) {
        throw std::invalid_argument(std::string("Given index is out of bound: ") + std::to_string(raw));
    }

    return Move(raw, board_size);
}

bool Move::operator==(Move const& other) const {
    if (raw_ < 0) {
        return raw_ == other.raw_;
    }
    else {
        return raw_ == other.raw_ and board_size_ == other.board_size_;
    }
}

bool Move::operator!=(Move const& other) const {
    return not (*this == other);
}

int Move::operator()() const {
    return raw_;
}

int Move::row() const {
    return raw_ / board_size_;
}

int Move::col() const {
    return raw_ % board_size_;
}

int Move::raw() const {
    return raw_;
}

int Move::board_size() const {
    return board_size_;
}

Move Move::n() const {
    if (col() == 0) {
        return INVALID;
    }

    return Move::from_coordinate(row(), col() - 1, board_size_);
}

Move Move::s() const {
    if (col() == board_size_ - 1) {
        return INVALID;
    }

    return Move::from_coordinate(row(), col() + 1, board_size_);;
}

Move Move::w() const {
    if (row() == 0) {
        return INVALID;
    }

    return Move::from_coordinate(row() - 1, col(), board_size_);
}

Move Move::e() const {
    if (row() == board_size_ - 1) {
        return INVALID;
    }

    return Move::from_coordinate(row() + 1, col(), board_size_);
}

bool Move::is_on_edge() const {
    int r = row();
    int c = col();

    return r == 0 or r == board_size_ - 1 or c == 0 or c == board_size_ - 1;
}

bool Move::is_at_corner() const {
    int r = row();
    int c = col();

    if (r == 0 and c == 0) {
        return true;
    }

    if (r == 0 and c == board_size_ - 1) {
        return true;
    }

    if (r == board_size_ - 1 and c == 0) {
        return true;
    }

    if (r == board_size_ - 1 and c == board_size_ - 1) {
        return true;
    }

    return false;
}

std::string Move::to_string() const {
    constexpr auto ROW = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

    std::stringstream ss;

    ss << ROW[col()] << (row() + 1);
    return ss.str();
}

}  // namespace cygo
