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
    add_executable(${test_name} ${srcs})
    add_dependencies(${test_name} cygo-core)

    target_link_libraries(${test_name}
        cygo-core
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(${test_name})

    add_test(
        NAME ${test_name}
        COMMAND $<TARGET_FILE:${test_name}>
    )
endif()
