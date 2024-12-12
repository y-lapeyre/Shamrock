

set(Sources
karras_alg.cpp
reduction_alg.cpp
compute_ranges.cpp
convert_ranges.cpp
)
set(SourcesTest

)
set(Headers
karras_alg.hpp
reduction_alg.hpp
compute_ranges.hpp
convert_ranges.hpp
)
set(HeadersTest

)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
        ${SourcesTest}
    PUBLIC
        ${HeadersTest}
    )
endif()
