#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "patchscheduler/scheduler_mpi.hpp"

template <class vectype, class primtype> class InterfaceHandler {

  private:
    std::vector<InterfaceComm<vectype>> interface_comm_list;

  public:
    template <class interface_selector>
    inline void compute_interface_list(SchedulerMPI &sched, SerialPatchTree<vectype> sptree, PatchField<primtype> h_field) {
        interface_comm_list = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
            sched, sptree, h_field, format("interfaces_%d_node%d", 0, mpi_handler::world_rank));
    }

    void comm_interfaces(SchedulerMPI &sched){
        SyCLHandler &hndl = SyCLHandler::get_instance();

        std::unordered_map<u64,std::vector<std::tuple<u64,std::unique_ptr<PatchData>>>> Interface_map;
        for (const Patch & p : sched.patch_list.global) {
            Interface_map[p.id_patch] = std::vector<std::tuple<u64,std::unique_ptr<PatchData>>>();
        }

        std::vector<std::unique_ptr<PatchData>> comm_pdat;
        std::vector<u64_2> comm_vec;

        

    }

    void get_interface_map();
};