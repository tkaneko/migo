set(srcs
    test_chain.cpp
    test_chain_group.cpp
    test_features.cpp
    test_move.cpp
    test_neighbor_counter.cpp
    test_state.cpp
    test_state_impl.cpp
    test_zobrist_hash.cpp
)
set(test_name cygo_test)


if(${CYGO_ENABLE_TESTING})
    include_directories(${GOOGLETEST_INCLUDE_DIR})
    link_directories(${GOOGLETEST_LIBRARY_DIR})

    add_executable(${test_name} ${srcs})

    add_dependencies(${test_name} cygo-core)
    add_dependencies(${test_name} googletest)

    target_link_libraries(${test_name}
        libgmock.a
        libgtest.a
        libgtest_main.a
        cygo-core
        pthread
    )

    add_test(
        NAME ${test_name}
        COMMAND $<TARGET_FILE:${test_name}>
    )
endif()