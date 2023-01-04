
# if("${BUILD_TEST}")

#     target_sources(shamrock_test
#         PRIVATE
#         PUBLIC
#             patch.hpp
#     )


# endif()

#add_subdirectory(local)

if("${BUILD_TEST}")

    target_sources(shamrock_test
        PRIVATE
            radix_tree.cpp
        PUBLIC
            radix_tree.hpp
    )


endif()


if("${BUILD_SIM}")

    target_sources(shamrock
        PRIVATE
            radix_tree.cpp
        PUBLIC
            radix_tree.hpp
    )


endif()

add_subdirectory(kernels)