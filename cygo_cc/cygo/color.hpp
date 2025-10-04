#ifndef CYGO_COLOR_HPP
#define CYGO_COLOR_HPP
#include <stdexcept>

namespace cygo {

enum class Color {
    BLACK = +1,
    WHITE = -1,
    EMPTY =  0
};

constexpr int is_player(Color c) { return c == Color::BLACK || c == Color::WHITE; }
constexpr int player_idx(Color c) { return (static_cast<int>(c)+1)/2; }
  
inline constexpr Color opposite_color(Color c) {
    if (c == Color::BLACK) {
        return Color::WHITE;
    }

    if (c == Color::WHITE) {
        return Color::BLACK;
    }

    return Color::EMPTY;
}

  template <class T>
  struct PlayerMap {
      T data[2];
      T& operator[](Color c) { return data[player_idx(c)]; }
      const T& operator[](Color c) const { return data[player_idx(c)]; }

      T& at(Color c) {
          if (! is_player(c))
              throw std::logic_error("color");
          return data[player_idx(c)];
      }
      const T& at(Color c) const {
          if (! is_player(c))
              throw std::logic_error("color");
          return data[player_idx(c)];
      }
  };

}  // namespace cygo

#endif //CYGO_COLOR_HPP
