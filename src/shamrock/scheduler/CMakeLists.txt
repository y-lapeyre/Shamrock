
set(Sources
PatchTree.cpp
HilbertLoadBalance.cpp
SchedulerPatchData.cpp
SerialPatchTree.cpp
        
scheduler_mpi.cpp 
scheduler_patch_list.cpp 
)
set(SourcesTest
    
)
set(Headers

)
set(HeadersTest

)


if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
     ${SourcesTest}
    PUBLIC
     ${HeadersTest}
    )
endif()
