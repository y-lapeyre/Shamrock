
set(Sources
    NodeInstance.cpp
    SyclMpiTypes.cpp
)
set(SourcesTest
    
)
set(Headers
    NodeInstance.hpp
    SyclMpiTypes.hpp
)
set(HeadersTest

)

if("${BUILD_SIM}")
target_sources(shamrock
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
        ${Sources} ${SourcesTest}
    PUBLIC
        ${Headers} ${HeadersTest}
    )
endif()

add_subdirectory(tests)
add_subdirectory(legacy)
add_subdirectory(experimental)
