
set(Sources

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


add_subdirectory(reduction)
add_subdirectory(numeric)
add_subdirectory(memory)
add_subdirectory(random)
add_subdirectory(algorithm)