#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cygo/color.hpp"
#include "cygo/move.hpp"
#include "cygo/state.hpp"

#include "cygo/binding/features.hpp"
#include "cygo/binding/miscs.hpp"

namespace py = pybind11;

using namespace pybind11::literals;


namespace std {

template <>
struct hash<std::pair<int, int>> {
    size_t operator()(std::pair<int, int> const& p) const {
        auto hasher = hash<int>();
        return hasher(p.first) ^ hasher(p.second);
    }
};

}


void setup_color(py::module& m) {
      py::enum_<cygo::Color>(m, "Color", "define color id following :cpp:enum:`cygo::Color`")
            .value("BLACK", cygo::Color::BLACK)
            .value("WHITE", cygo::Color::WHITE)
            .value("EMPTY", cygo::Color::EMPTY)
            .export_values()
      .def("opponent",
           &cygo::opposite_color,
           "Return the opposite color\n\n"
           ">>> cygo.BLACK.opponent() == cygo.WHITE\n"
           "True\n"
           )
      ;
}

void setup_move(py::module& m) {
    py::class_<cygo::Move>(m, "Move")
            .def_static("from_coordinate",
                 &cygo::Move::from_coordinate,
                 "construct Move object from coordinate and board size\n\n"
                 ">>> move = cygo.Move.from_coordinate(1, 2, board_size=4)\n"
                 ">>> move\n"
                 "(1, 2)\n"
                 ">>> move.row\n"
                 "1\n"
                 ">>> move.col\n"
                 "2\n"
                 ">>> move.board_size\n"
                 "4\n"
                 ">>> move.raw()\n"
                 "6\n"
                 ">>> move.n\n"
                 "(1, 1)\n"
                 ">>> move.s\n"
                 "(1, 3)\n"
                 ">>> move.w\n"
                 "(0, 2)\n"
                 ">>> move.e\n"
                 "(2, 2)\n",
                 "row"_a, "col"_a, "board_size"_a
            )
            .def_static("from_gtp_string", &cygo::Move::from_gtp_string,
                 "gtp_string"_a, "board_size"_a,
                 "construct Move object from gtp string and board size\n\n"
                 ">>> a1 = cygo.Move.from_gtp_string('a1', board_size=4)\n"
                 ">>> a1\n"
                 "(0, 0)\n"
                 ">>> a1.is_at_corner\n"
                 "True\n"
                 ">>> a1.is_on_edge\n"
                 "True\n"
                 ">>> c2 = cygo.Move.from_gtp_string('c2', 4)\n"
                 ">>> c2\n"
                 "(1, 2)\n"
                 ">>> c2.is_on_edge\n"
                 "False\n"
                 ">>> c2.gtp\n"
                 "'C2'\n"
                 ">>> pass_move = cygo.Move.from_gtp_string('pass', 4)\n"
                 ">>> pass_move\n"
                 "Move.Pass\n"
                 ">>> pass_move.gtp\n"
                 "'PASS'\n"
            )
            .def_static("from_raw_value", &cygo::Move::from_raw,
                 "raw_value"_a, "board_size"_a,
                 "construct Move object from internal representation\n\n"
                 ">>> move = cygo.Move.from_raw_value(6, board_size=4)\n"
                 ">>> move\n"
                 "(1, 2)\n"
            )
            .def("raw", &cygo::Move::raw)
            .def_property_readonly("row", &cygo::Move::row, "row in int")
            .def_property_readonly("col", &cygo::Move::col, "col in int")
            .def_property_readonly("board_size", &cygo::Move::board_size, "board_size in int")
            .def_property_readonly("n", &cygo::Move::n, "move at neighbor")
            .def_property_readonly("s", &cygo::Move::s, "move at neighbor")
            .def_property_readonly("w", &cygo::Move::w, "move at neighbor")
            .def_property_readonly("e", &cygo::Move::e, "move at neighbor")
            .def_property_readonly("is_on_edge", &cygo::Move::is_on_edge, "true if on edge")
            .def_property_readonly("is_at_corner", &cygo::Move::is_at_corner,
                                   "true if on any of four corners")
            .def_property_readonly("is_pass", [](cygo::Move const& v) -> bool {
                return v == cygo::Move::PASS;
            }, "true if pass")
            .def_property_readonly("gtp", [](cygo::Move const& v) -> std::string {
                if (v == cygo::Move::PASS) {
                    return "PASS";
                }
                std::ostringstream os;
                int col = v.col();
                if (col >= 8)
                  ++col;
                os << char('A' + col) << (v.row() + 1);
                return os.str();
            }, "gtp representation")
            .def("__repr__", [](cygo::Move const& v) -> std::string {
                if (v == cygo::Move::PASS) {
                    return "Move.Pass";
                }

                std::ostringstream os;
                os << "(" << v.row() << ", " << v.col() << ")";

                return os.str();
            })
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def_property_readonly_static("Pass", [](py::object) { return cygo::Move::PASS; });
}


void setup_state(py::module& m) {
    py::class_<cygo::State>(m, "State",
                "Go state\n\n"
                ":param board_size: int\n"
                ":param komi: float = 7.5\n"
                ":param superko_rule: bool = True\n"
                ":param max_history_n: int = 7\n\n"
                ">>> state = cygo.State(4)\n"
                ">>> print(state)\n"
                "  A  B  C  D \n"
                "4 .  .  .  . 4\n"
                "3 .  .  .  . 3\n"
                "2 .  .  .  . 2\n"
                "1 .  .  .  . 1\n"
                "  A  B  C  D \n"
                "...\n"
                ">>> state.make_move((2, 1))\n"
                ">>> print(state)\n"
                "  A  B  C  D \n"
                "4 .  .  .  . 4\n"
                "3 . (X) .  . 3\n"
                "2 .  .  .  . 2\n"
                "1 .  .  .  . 1\n"
                "  A  B  C  D \n"
                "...\n"
                ">>> state.make_move('c3')\n"
                ">>> print(state)\n"
                "  A  B  C  D \n"
                "4 .  .  .  . 4\n"
                "3 .  X (O) . 3\n"
                "2 .  .  .  . 2\n"
                "1 .  .  .  . 1\n"
                "  A  B  C  D \n"
                "...\n"
                ">>> state.last_move\n"
                "(2, 2)\n"
            )
            .def(py::init<std::size_t, double, bool, int>(),
                 "board_size"_a, "komi"_a = 7.5, "superko_rule"_a = true, "max_history_n"_a = 7
            )
            .def_property("current_player",
                          [](cygo::State const& state) { return state.current_player; },
                          [](cygo::State& state, cygo::Color c) { state.current_player = c; },
                          "Opposite of the last played color\n\n" ":type: :py:class:`cygo.Color`\n"
            )
            .def_property("komi",
                          [](cygo::State const& state) { return state.komi; },
                          [](cygo::State& state, double komi) { state.komi = komi; },
                          "Komi value\n\n" ":type: float\n"
            )
            .def_property_readonly("board_size",
                                   [](cygo::State const& state) { return state.board_size(); },
                                   "Return the current board size\n\n" ":type: int\n"
            )
            .def_property_readonly("last_move",
                                   [](cygo::State const& state) { return state.last_move(); },
                                   "Return the last move\n\n" ":type: :py:class:`cygo.Move`\n"
            )
            .def_property_readonly("max_history_n",
                                   [](cygo::State const& state) { return state.max_history_n(); },
                                   "maximum length of history to remember\n\n" ":type: int\n"
            )
            .def_property_readonly("superko_rule",
                                   [](cygo::State const& state) { return state.superko_rule(); },
                                   "True if superko is adopted\n\n" ":type: bool\n"
            )
            .def_property_readonly("zobrist_hash",
                                   [](cygo::State const& state) { return state.hash(); },
                                   "return 64bit hash for board status (ignoring history) to quickly detect equivalence\n\n"
                                   ">>> state = cygo.State(5)\n"
                                   ">>> moves = ['B3', 'C2', 'C4']\n"
                                   ">>> print(state.zobrist_hash)\n"
                                   "0\n"
                                   ">>> for move in moves:\n"
                                   "...   state.make_move(move)\n"
                                   "...   print(state.zobrist_hash)\n"
                                   "12092317580524320504\n"
                                   "13032588549984992753\n"
                                   "493738324825164472\n"
                                   ">>> state2 = cygo.State(5)\n"
                                   ">>> for move in reversed(moves):\n"
                                   "...   state2.make_move(move)\n"
                                   ">>> state.zobrist_hash == state2.zobrist_hash\n"
                                   "True\n"
            )
            .def("__str__",
                 [] (cygo::State const& state) {
                     return state.to_string();
                 }
            )
            .def("copy",
                 [] (cygo::State const& state) {
                     return cygo::State(state);
                 },
                 py::return_value_policy::move
            )
            .def("legal_moves",
                 [] (cygo::State const& state, cygo::Color c, bool include_eyeish) {
                     auto legals = state.legal_moves(c, include_eyeish);

                     std::unordered_set<std::pair<int, int>> ret;
                     for (auto const& v : legals) {
                         ret.emplace(std::make_pair(v.row(), v.col()));
                     }

                     return ret;
                 },
                 "Generate legal moves for the current state",
                 "color"_a = cygo::Color::EMPTY, "include_eyeish"_a = true
            )
            .def("legal_indices",
                 [] (cygo::State const& state, cygo::Color c, bool include_eyeish) {
                     auto legals = state.legal_moves(c, include_eyeish);

                     std::vector<int> ret(legals.size());

                     int i = 0;
                     for (auto const& v : legals) {
                         ret[i] = v.raw();
                         ++ i;
                     }

                     return ret;
                 },
                 "Generate legal indices for the current state",
                 "color"_a = cygo::Color::EMPTY, "include_eyeish"_a = true
            )
            .def("is_legal",
                 [](cygo::State const& state, cygo::Move const* m, cygo::Color c) {
                     if (m == nullptr) {
                         return true;
                     }

                     return state.is_legal(*m, c);
                 },
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("is_legal",
                 [](cygo::State const& state, std::string const& m, cygo::Color c) {
                     return state.is_legal(cygo::Move::from_gtp_string(m, state.board_size()), c);
                 },
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("is_legal",
                 [](cygo::State const& state, std::pair<int, int> const& v, cygo::Color c) {
                     return state.is_legal(cygo::Move::from_coordinate(v.first, v.second, state.board_size()), c);
                 },
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("tromp_taylor_score",
                 &cygo::State::tromp_taylor_score,
                 "Returns tromp taylor score from Color perspective with :py:attr:`cygo.State.komi`\n\n"
                 ":param color: viewpoint (EMPTY for turn)\n",
                 ">>> state = cygo.State(5, komi=1)\n"
                 ">>> moves = ['a2', 'a3', 'b2', 'b3', 'c2', 'c3', 'd2', 'd3', 'e2', 'e3']\n"
                 ">>> for m in moves:\n"
                 "...   state.make_move(m)\n"
                 ">>> print(state)\n"
                 "  A  B  C  D  E \n"
                 "5 .  .  .  .  . 5\n"
                 "4 .  .  .  .  . 4\n"
                 "3 O  O  O  O (O)3\n"
                 "2 X  X  X  X  X 2\n"
                 "1 .  .  .  .  . 1\n"
                 "  A  B  C  D  E \n"
                 "...\n"
                 ">>> state.tromp_taylor_score(cygo.Color.BLACK)\n"
                 "-6.0\n"
                 ">>> state.tromp_taylor_score()\n"
                 "-6.0\n"
                 ">>> state.make_move('C1')\n"
                 ">>> print(state)\n"
                 "  A  B  C  D  E \n"
                 "5 .  .  .  .  . 5\n"
                 "4 .  .  .  .  . 4\n"
                 "3 O  O  O  O  O 3\n"
                 "2 X  X  X  X  X 2\n"
                 "1 .  . (X) .  . 1\n"
                 "  A  B  C  D  E \n"
                 "...\n"
                 ">>> state.tromp_taylor_score(cygo.Color.BLACK)\n"
                 "-6.0\n"
                 ">>> state.tromp_taylor_score()\n"
                 "6.0\n"
                 ">>> state.make_move('D1')\n"
                 ">>> print(state)\n"
                 "  A  B  C  D  E \n"
                 "5 .  .  .  .  . 5\n"
                 "4 .  .  .  .  . 4\n"
                 "3 O  O  O  O  O 3\n"
                 "2 X  X  X  X  X 2\n"
                 "1 .  .  X (O) . 1\n"
                 "  A  B  C  D  E \n"
                 "...\n"
                 ">>> state.tromp_taylor_score(cygo.Color.BLACK)\n"
                 "-9.0\n"
                 ">>> state.make_move('E1')\n"
                 ">>> print(state)\n"
                 "  A  B  C  D  E \n"
                 "5 .  .  .  .  . 5\n"
                 "4 .  .  .  .  . 4\n"
                 "3 O  O  O  O  O 3\n"
                 "2 X  X  X  X  X 2\n"
                 "1 .  .  X  . (X)1\n"
                 "  A  B  C  D  E \n"
                 "...\n"
                 ">>> state.tromp_taylor_score(cygo.Color.BLACK)\n"
                 "-6.0\n",
                 "color"_a = cygo::Color::EMPTY
            )
            .def_property_readonly("move_history", [] (cygo::State& state) {
              return state.move_history(cygo::Color::EMPTY);
            }, "Returns color's move history list"
            )
            .def("color_move_history",
                 &cygo::State::move_history,
                 "Returns color's move history list",
                 "color"_a
            )
            .def("make_move",
                 [] (cygo::State& state, int index, cygo::Color c) {
                     if (index == -1) {
                         state.make_move(cygo::Move::PASS, c);
                     }
                     else {
                         state.make_move(cygo::Move::from_raw(index, state.board_size()), c);
                     }
                 },
                 "Apply move to the state as color\n\n"
                 ":param index: index acceptable by :cpp:func:`cygo::Move::from_raw` or -1 for pass\n",
                 "index"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("make_move",
                 [] (cygo::State& state, cygo::Move const* m, cygo::Color c) {
                     if (m == nullptr) {
                         state.make_move(cygo::Move::PASS, c);
                     }
                     else {
                         state.make_move(*m, c);
                     }
                 },
                 "Apply move to the state as color\n\n"
                 ":param move: None for pass\n",
                 "move"_a.none(true), "color"_a = cygo::Color::EMPTY
            )
            .def("make_move",
                 [] (cygo::State& state, std::string const& move, cygo::Color c) {
                     state.make_move(cygo::Move::from_gtp_string(move, state.board_size()), c);
                 },
                 "Apply move to the state as color\n\n",
                 ":param move: string acceptable by :py:meth:`cygo.Move.from_gtp_string`\n",
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("make_move",
                 [] (cygo::State& state, std::pair<int, int> const& v, cygo::Color c) {
                    state.make_move(cygo::Move::from_coordinate(v.first, v.second, state.board_size()), c);
                 },
                 "Apply move to the state as color",
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("color_at",
                 [] (cygo::State& state, std::pair<int, int> const& v) {
                     const auto& stones = state.stones();
                     auto m = cygo::Move::from_coordinate(v.first, v.second, state.board_size());
                     return stones.at(m());
                 },
                 "return cygo::Color at vertex",
                 "vertex"_a
            );
      ;
}

void setup_attributes(py::module& m) {
    m.attr("Pass") = nullptr;

    m.def("apply_moves",
          &cygo::apply_moves,
          "Apply given moves to the given state.\n\n"
          "The moves should be an ndarray of ints, "
          "each of which is Move.raw or -1 for pass.\n\n"
          ".. warning:: -1 is inconsistent with :cpp:member:`cygo::Move::PASS` which is -2",
          "state"_a, "moves"_a
    );

    m.def("zobrist_hash",
          &cygo::zobrist_hash,
          "Calculate zobrist hash for a given position represented by two ndarrays",
          "black_array"_a, "white_array"_a
    );

    m.def("opposite_color",
          &cygo::opposite_color,
          "Return the opposite color of a given color",
          "color"_a
    );
}

void setup_features(py::module& m) {
    py::module m_features = m.def_submodule("features");

    m_features.def("board_i",
                   [] (cygo::State const& state, int i, cygo::Color c) {
                       return cygo::board_i(state, i, c);
                   },
                   "Get the i-th before board feature. If c = Color.EMPTY, returns both color's features",
                   "state"_a, "i"_a, "color"_a = cygo::Color::EMPTY
    );
    m_features.def("history_n",
                   [] (cygo::State const& state, int n, cygo::Color c) {
                       return cygo::history_n(state, n, c);
                   },
                   "Get the board history from n-th before. If c = Color.EMPTY, returns both color's features",
                   "state"_a, "i"_a, "color"_a = cygo::Color::EMPTY
    );

    m_features.def("color_black", &cygo::black);
    m_features.def("color_white", &cygo::white);
}


PYBIND11_MODULE(cygo, m) {
    setup_color(m);
    setup_move(m);
    setup_state(m);
    setup_attributes(m);
    setup_features(m);
}

