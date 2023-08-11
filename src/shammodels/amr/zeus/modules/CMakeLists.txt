
set(Sources
GhostZones.cpp
AMRTree.cpp
ComputePressure.cpp
FaceFlagger.cpp
SourceStep.cpp
DiffOperator.cpp
)
set(Headers
)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
