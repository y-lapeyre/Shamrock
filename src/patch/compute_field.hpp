#pragma once

#include "aliases.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include <memory>
#include <unordered_map>
#include <vector>





template<class T>
class PatchComputeField{public:

    std::unordered_map<u64, std::vector<T>> field_data;


    inline void generate(SchedulerMPI & sched){

        sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
            field_data[id_patch].resize(pdat_buf.element_count);
            sycl::buffer<T> field_buf(field_data[id_patch].data(),field_data[id_patch].size());
        });

    }

    std::unordered_map<u64, std::unique_ptr<sycl::buffer<T>>> field_data_buf;
    inline void to_sycl(){
        for (auto & [key,dat] : field_data) {
            field_data_buf[key] = std::make_unique<sycl::buffer<T>>(dat.data(),dat.size());
        }
    }

    inline void to_map(){
        field_data_buf.clear();
    }



};

template<class T>
class PatchComputeFieldInterfaces{public:

    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<std::vector<T>>>>> interface_map;





};