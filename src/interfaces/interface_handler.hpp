#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "patchscheduler/scheduler_mpi.hpp"

template <class vectype, class primtype> class InterfaceHandler {

  private:
    std::vector<InterfaceComm<vectype>> interface_comm_list;
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> interface_map;

  public:
    template <class interface_selector>
    inline void compute_interface_list(SchedulerMPI &sched, SerialPatchTree<vectype> sptree, PatchField<primtype> h_field) {
        interface_comm_list = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
            sched, sptree, h_field, format("interfaces_%d_node%d", 0, mpi_handler::world_rank));
    }

    void comm_interfaces(SchedulerMPI &sched);

    inline const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> &get_interface_list(u64 key) {
        return interface_map[key];
    }

    inline void print_current_interf_map() {

        for (const auto &[pid, int_vec] : interface_map) {
            printf(" pid : %d :\n", pid);
            for (auto &[a, b] : int_vec) {
                printf("    -> %d : len %d\n", a, b->pos_s.size() + b->pos_d.size());
            }
        }
    }
};