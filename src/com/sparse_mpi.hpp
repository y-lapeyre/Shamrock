#pragma once

#include "aliases.hpp"


struct MPI_Packet{
    u32 rank;
    std::vector<u8> patchdata;
};