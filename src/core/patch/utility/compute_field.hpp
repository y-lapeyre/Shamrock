// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/patch/base/patchdata_field.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include <memory>
#include <unordered_map>
#include <vector>




template<class T>
class PatchComputeField{public:

    std::unordered_map<u64, PatchDataField<T>> field_data;


    inline void generate(PatchScheduler & sched){
        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
            field_data.insert({id_patch,PatchDataField<T>("comp_field",1)});
            field_data.at(id_patch).resize(pdat_buf.element_count);
        });
    }

    
    inline void generate(PatchScheduler & sched, std::unordered_map<u64, u32>& size_map){
        sched.for_each_patch([&](u64 id_patch, Patch cur_p) {
            field_data.insert({id_patch,PatchDataField<T>("comp_field",1)});
            field_data.at(id_patch).resize(size_map[id_patch]);
        });
    }

    std::unordered_map<u64, std::unique_ptr<sycl::buffer<T>>> field_data_buf;
    inline void to_sycl(){
        for (auto & [key,dat] : field_data) {
            field_data_buf[key] = std::make_unique<sycl::buffer<T>>(dat.usm_data(),dat.size());
        }
    }

    inline void to_map(){
        field_data_buf.clear();
    }



};

template<class T>
class PatchComputeFieldInterfaces{public:

    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>>> interface_map;





};