set(srcs
    features.cpp
    main.cpp
    miscs.cpp
)

set(module_name cygo)

pybind11_add_module(${module_name} MODULE ${srcs})

add_dependencies(${module_name} cygo-core)

target_link_libraries(${module_name} PRIVATE cygo-core)
