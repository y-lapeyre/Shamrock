
set(Sources
GhostZones.cpp
AMRTree.cpp
ComputePressure.cpp
FaceFlagger.cpp
SourceStep.cpp
)
set(Headers
)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
