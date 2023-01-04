
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
        test_mpi_scheduler.cpp
        test_patchdata.cpp
        test_hilbert_sfc.cpp
        test_patchtree.cpp
        test_scheduler_patch_list.cpp
    PUBLIC
         
    )


endif()
