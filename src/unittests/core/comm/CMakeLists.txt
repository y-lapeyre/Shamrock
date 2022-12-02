
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_sparse_communicator_patchdata.cpp
        test_sparse_communicator_patchdata_field.cpp
        test_sparse_communicator_radix_tree.cpp
    PUBLIC
         
    )


endif()
