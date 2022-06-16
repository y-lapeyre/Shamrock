// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file interface_handler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "io/logs.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sph/sphpatch.hpp"

#include "interface_handler_impl.hpp"

/**
 * @brief 
 * 
 * @tparam vectype 
 * @tparam primtype 
 * //TODO check that for periodic BC case : if a patch has itself as interface, is there any bug because of map {map } repr
 * //TODO put the flag choice thing in a separate function to avoid recomputation
 */
template <class vectype, class primtype> class InterfaceHandler {

  private:
    /**
     * @brief 
     * 
     */
    std::vector<InterfaceComm<vectype>> interface_comm_list;

    /**
     * @brief 
     * 
     */
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> interface_map;

  public:

    /**
     * @brief 
     * 
     * @tparam interface_selector 
     * @param sched 
     * @param sptree 
     * @param h_field 
     */
    template <class interface_selector>
    inline void compute_interface_list(PatchScheduler &sched, SerialPatchTree<vectype> sptree, PatchField<primtype> h_field,bool periodic) {
        auto t = timings::start_timer("compute_interface_list", timings::function);
        interface_comm_list = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
            sched, sptree, h_field, format("interfaces_%d_node%d", 0, mpi_handler::world_rank),periodic);
        t.stop();
    }

    /**
     * @brief 
     * 
     * @param sched 
     */
    void comm_interfaces(PatchScheduler &sched,bool periodic);


    template <class T> PatchComputeFieldInterfaces<T> comm_interfaces_field(PatchScheduler &sched,PatchComputeField<T> &pcomp_field,bool periodic) {

        PatchComputeFieldInterfaces<T> interface_field_map;

        auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
        impl::comm_interfaces_field<T,vectype>(sched, pcomp_field, interface_comm_list, interface_field_map.interface_map, periodic);
        t.stop();

        return interface_field_map;
    }

    

    /**
     * @brief Get the interface list object
     * 
     * @param key 
     * @return const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>& 
     */
    inline const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> &get_interface_list(u64 key) {
        return interface_map[key];
    }


    

    /**
     * @brief 
     * 
     */
    inline void print_current_interf_map() {

        for (const auto &[pid, int_vec] : interface_map) {
            printf(" pid : %d :\n", pid);
            for (auto &[a, b] : int_vec) {
                //printf("    -> %d : len %d\n", a, b->obj_cnt);
            }
        }
    }



    template<class Function>
    inline void for_each_interface(u64 patch_id,sycl::queue & queue, Function && fct){

        const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst = get_interface_list(patch_id);

        for (auto & [int_pid, pdat_ptr] : p_interf_lst) {

            if(! pdat_ptr->is_empty()){

                PatchDataBuffer pdat_buf = attach_to_patchData(*pdat_ptr);

                auto t = patchdata::sph::get_patchdata_BBAA<vectype>(queue, pdat_buf);

                fct(patch_id,int_pid,pdat_buf,t);
            }
        }

    }

};



template <> void InterfaceHandler<f32_3, f32>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}

template <> void InterfaceHandler<f64_3, f64>::comm_interfaces(PatchScheduler &sched,bool periodic) {
    auto t = timings::start_timer("comm interfaces", timings::timingtype::function);
    impl::comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map,periodic);
    t.stop();
}









