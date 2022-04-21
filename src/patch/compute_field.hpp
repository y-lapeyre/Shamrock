#pragma once

#include "aliases.hpp"
#include "hipSYCL/sycl/buffer.hpp"
#include <unordered_map>
#include <vector>





template<class T>
class PatchComputeField{public:

    std::unordered_map<u64, std::vector<T>> field_data;

    template<class Function>
    inline void generate(u64 patch_id, u32 obj_cnt , Function && lambda){

        field_data[patch_id].resize(obj_cnt);
        sycl::buffer<T> field_buf(field_data[patch_id].data(),field_data[patch_id].size());

        lambda(field_buf);

    }





};

template<class T>
class PatchComputeFieldInterfaces{public:

    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<std::vector<T>>>>> interface_map;





};