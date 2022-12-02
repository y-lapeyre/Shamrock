
if("${BUILD_SIM}")

target_sources(shamrock
    PRIVATE
        #scheduler.cpp
        
        
        
        sparse_communicator_patchdata.cpp
        sparse_communicator_patchdata_field.cpp
        sparse_communicator_radix_tree.cpp
        sparse_communicator_rtree_buf.cpp
        #patchdata_buffer.cpp 


        
    PUBLIC
        #patch.hpp
        
        #patchdata.hpp
        #scheduler.hpp
    )

endif()


if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE

        
    sparse_communicator_patchdata.cpp
    sparse_communicator_patchdata_field.cpp
    sparse_communicator_radix_tree.cpp
    sparse_communicator_rtree_buf.cpp
    #patchdata_buffer.cpp 


    PUBLIC
        #patch.hpp
        #patchdata.hpp
        #scheduler.hpp
    )


endif()
