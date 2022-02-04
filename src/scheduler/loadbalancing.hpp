#pragma once

#include "../aliases.hpp"

#include "patch.hpp"

#include <vector>
#include <tuple>

inline std::vector<std::tuple<u32, i32, i32,i32>> make_change_list(std::vector<Patch> & global_patch_list){

    std::vector<std::tuple<u32, i32, i32,i32>> change_list;


    std::vector<u64> hilberts_code(global_patch_list.size());
    std::vector<u64> patch_load(global_patch_list.size());


    





    return change_list;
}