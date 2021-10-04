#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include <vector>

//this one is mandatory
const bool use_field_r = true;

bool use_field_rho = false;
bool use_field_v = false;


class PatchData{

    

};

class Patch{

    u64_3 patch_pos_min;
    u64_3 patch_pos_max;

    PatchData* data;

};