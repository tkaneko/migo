#ifndef CYGO_COLOR_HPP
#define CYGO_COLOR_HPP


namespace cygo {

enum class Color {
    BLACK = +1,
    WHITE = -1,
    EMPTY =  0
};

inline constexpr Color opposite_color(Color c) {
    if (c == Color::BLACK) {
        return Color::WHITE;
    }

    if (c == Color::WHITE) {
        return Color::BLACK;
    }

    return Color::EMPTY;
}

}  // namespace cygo

#endif //CYGO_COLOR_HPP
