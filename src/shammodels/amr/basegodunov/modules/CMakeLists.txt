
set(Sources
    GhostZones.cpp
    AMRTree.cpp
    StencilGenerator.cpp
    AMRGraphGen.cpp
    ComputeGradient.cpp
)
set(Headers
)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
