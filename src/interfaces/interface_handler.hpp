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
#include "patchscheduler/scheduler_mpi.hpp"

/**
 * @brief 
 * 
 * @tparam vectype 
 * @tparam primtype 
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
    inline void compute_interface_list(SchedulerMPI &sched, SerialPatchTree<vectype> sptree, PatchField<primtype> h_field) {
        auto t = timings::start_timer("compute_interface_list", timings::function);
        interface_comm_list = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
            sched, sptree, h_field, format("interfaces_%d_node%d", 0, mpi_handler::world_rank));
        t.stop();
    }

    /**
     * @brief 
     * 
     * @param sched 
     */
    void comm_interfaces(SchedulerMPI &sched);

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
                printf("    -> %d : len %d\n", a, b->pos_s.size() + b->pos_d.size());
            }
        }
    }
};