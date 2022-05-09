#pragma once


#include "patchdata_exchanger_impl.hpp"
#include <vector>


inline void patch_data_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchData>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> & interface_map
    ){
        patchdata_exchanger::impl::patch_data_exchange_object(global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
    }

template<class T>
inline void patch_data_field_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<std::vector<T>>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<std::vector<T>>>>> & interface_map
    ){
        patchdata_exchanger::impl::patch_data_field_exchange_object(global_patch_list, send_comm_pdat, send_comm_vec, interface_map);
    }
