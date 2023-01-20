
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_key_pair_sort.cpp
    PUBLIC
         
    )


endif()
