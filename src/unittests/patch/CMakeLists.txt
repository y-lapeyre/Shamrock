
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        patch_reduc_tree_test.cpp
        #test_patchdata_field.cpp
        test_patchdata.cpp
    PUBLIC
         
    )


endif()
