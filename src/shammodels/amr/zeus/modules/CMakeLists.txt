
set(Sources
GhostZones.cpp
AMRTree.cpp
ComputePressure.cpp
FaceFlagger.cpp
)
set(Headers
)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
