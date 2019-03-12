#include <ostream>
#include <stdexcept>

#include "cygo/state.hpp"
#include "cygo/features.hpp"

#include "cygo/binding/features.hpp"


namespace py = pybind11;

namespace cygo {

template <typename T, typename TInt>
py::array_t<T> make_pyarray(T* array_ptr, std::initializer_list<TInt> shape) {
    return py::array_t<T>(shape, array_ptr, py::capsule(array_ptr, [](void* f) {
        delete[] reinterpret_cast<T*>(f);
    }));
}

template <typename T>
T* to_ptr(std::vector<T> const& v) {
    auto ptr = new T[v.size()]();

    std::copy(std::begin(v), std::end(v), ptr);

    return ptr;
}

py::array_t<float> black(State const& state) {
    return make_pyarray(to_ptr(feature_impl::color<float>(state, Color::BLACK)),
                        {1, state.board_size(), state.board_size()});
}

py::array_t<float> white(State const& state) {
    return make_pyarray(to_ptr(feature_impl::color<float>(state, Color::WHITE)),
                        {1, state.board_size(), state.board_size()});
}


py::array_t<float> board_i(State const& state, int i, Color c) {
    auto const length = state.history().history_length;

    if (i < 0 or state.history().history_length <= i) {
        std::ostringstream os;
        os << "Given index (" << i << ") is out of bound. Expected in [0, " << length << ")";

        throw std::invalid_argument(os.str());
    }

    if (c == Color::EMPTY) {
        return make_pyarray(to_ptr(feature_impl::board_i<float>(state, i)),
                            {2, state.board_size(), state.board_size()});
    }
    else {
        return make_pyarray(to_ptr(feature_impl::board_i_color<float>(state, i, c)),
                            {1, state.board_size(), state.board_size()});
    }
}

py::array_t<float> history_n(State const& state, int n, Color c) {
    auto const length = state.history().history_length;

    if (n < 0 or length <= n) {
        std::ostringstream os;
        os << "Given index (" << n << ") is out of bound. Expected in [0, " << length << ")";

        throw std::invalid_argument(os.str());
    }

    if (c == Color::EMPTY) {
        return make_pyarray(to_ptr(feature_impl::history_n<float>(state, n)),
                            {2 * (n + 1), state.board_size(), state.board_size()});
    }
    else {
        return make_pyarray(to_ptr(feature_impl::history_n_color<float>(state, n, c)),
                            {n + 1, state.board_size(), state.board_size()});
    }
}

}  // namespace cygo
