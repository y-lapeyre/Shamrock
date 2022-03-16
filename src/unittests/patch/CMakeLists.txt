
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        patch_reduc_tree_test.cpp
    PUBLIC
         
    )


endif()
