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
    py::enum_<cygo::Color>(m, "Color")
            .value("BLACK", cygo::Color::BLACK)
            .value("WHITE", cygo::Color::WHITE)
            .value("EMPTY", cygo::Color::EMPTY)
            .export_values();
}

void setup_move(py::module& m) {
    py::class_<cygo::Move>(m, "Move")
            .def_static("from_coordinate",
                 &cygo::Move::from_coordinate,
                 "construct Move object from coordinate and board size",
                 "row"_a, "col"_a, "board_size"_a
            )
            .def_static("from_gtp_string", &cygo::Move::from_gtp_string)
            .def("raw", &cygo::Move::raw)
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
    py::class_<cygo::State>(m, "State")
            .def(py::init<std::size_t, double, bool, int>(),
                 "board_size"_a, "komi"_a = 7.5, "superko_rule"_a = true, "max_history_n"_a = 7
            )
            .def_property("current_player",
                          [](cygo::State const& state) { return state.current_player; },
                          [](cygo::State& state, cygo::Color c) { state.current_player = c; },
                          "Opposite of the last played color"
            )
            .def_property("komi",
                          [](cygo::State const& state) { return state.komi; },
                          [](cygo::State& state, double komi) { state.komi = komi; },
                          "Komi value"
            )
            .def_property_readonly("board_size",
                                   [](cygo::State const& state) { return state.board_size(); },
                                   "Return the current board size"
            )
            .def_property_readonly("last_move",
                                   [](cygo::State const& state) { return state.last_move(); },
                                   "Return the last move"
            )
            .def_property_readonly("max_history_n",
                                   [](cygo::State const& state) { return state.max_history_n(); }
            )
            .def_property_readonly("superko_rule",
                                   [](cygo::State const& state) { return state.superko_rule(); }
            )
            .def_property_readonly("zobrist_hash",
                                   [](cygo::State const& state) { return state.hash(); }
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
                 "Returns tromp taylor score from Color perspective",
                 "color"_a = cygo::Color::EMPTY
            )
            .def("move_history",
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
                 "Apply move to the state as color",
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
                 "Apply move to the state as color",
                 "move"_a.none(true), "color"_a = cygo::Color::EMPTY
            )
            .def("make_move",
                 [] (cygo::State& state, std::string const& move, cygo::Color c) {
                     state.make_move(cygo::Move::from_gtp_string(move, state.board_size()), c);
                 },
                 "Apply move to the state as color",
                 "move"_a, "color"_a = cygo::Color::EMPTY
            )
            .def("make_move",
                 [] (cygo::State& state, std::pair<int, int> const& v, cygo::Color c) {
                    state.make_move(cygo::Move::from_coordinate(v.first, v.second, state.board_size()), c);
                 },
                 "Apply move to the state as color",
                 "move"_a, "color"_a = cygo::Color::EMPTY
            );
}

void setup_attributes(py::module& m) {
    m.attr("Pass") = nullptr;

    m.def("apply_moves",
          &cygo::apply_moves,
          "Apply given moves to the given state. The moves should be an ndarray of ints",
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

