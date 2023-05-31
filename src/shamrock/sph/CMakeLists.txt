
set(Sources
SPHUtilities.cpp
)
set(Headers
SPHUtilities.hpp
)

if("${BUILD_SIM}")
target_sources(shamrock_lib
PRIVATE
    ${Sources}
PUBLIC
    ${Headers}
)
endif()