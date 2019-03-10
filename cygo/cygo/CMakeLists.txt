set(cygo_srcs
    chain.cpp
    chain_group.cpp
    move.cpp
    neighbor_counter.cpp
    state.cpp
    state_impl.cpp
    zobrist_hash.cpp
)

add_library(cygo-core ${cygo_srcs})

set_target_properties(cygo-core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(binding)
add_subdirectory(tests)
