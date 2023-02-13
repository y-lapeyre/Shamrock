
set(Sources
algorithm.cpp
)
set(SourcesTest
    
)
set(Headers

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

add_subdirectory(details)
add_subdirectory(tests)