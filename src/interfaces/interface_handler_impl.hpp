#pragma once

#include "patch/compute_field.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "interface_generator.hpp"
#include "patch/patchdata_exchanger.hpp"
#include <vector>

namespace impl {
    
    template <class vectype, class primtype>
    void comm_interfaces(SchedulerMPI &sched, std::vector<InterfaceComm<vectype>> &interface_comm_list,
                        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> &interface_map,bool periodic) {
        SyCLHandler &hndl = SyCLHandler::get_instance();

        interface_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_map[p.id_patch] = std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>();
        }

        auto t1 = timings::start_timer("generate interfaces", timings::timingtype::function);
        std::vector<std::unique_ptr<PatchData>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0) {
                    std::vector<std::unique_ptr<PatchData>> pret = InterfaceVolumeGenerator::append_interface<vectype>(
                        hndl.get_queue_alt(0), sched.patch_data.owned_data[interface_comm_list[i].sender_patch_id],
                        {interface_comm_list[i].interf_box_min}, {interface_comm_list[i].interf_box_max},interface_comm_list[i].interf_offset);
                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PatchData>());
                }
                comm_vec.push_back(
                    u64_2{interface_comm_list[i].global_patch_idx_send, interface_comm_list[i].global_patch_idx_recv});
            }

            //std::cout << "\n split \n";
        }
        t1.stop();

        auto t2 = timings::start_timer("patch_data_exchange_object", timings::timingtype::mpi);
        patch_data_exchange_object(sched.patch_list.global, comm_pdat,comm_vec,interface_map);
        t2.stop();
    }




    template <class T,class vectype>
    void comm_interfaces_field(SchedulerMPI &sched, PatchComputeField<T> &pcomp_field, std::vector<InterfaceComm<vectype>> &interface_comm_list,
                        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<std::vector<T>>>>> &interface_field_map,bool periodic) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        using PCField = std::vector<T>;

        interface_field_map.clear();
        for (const Patch &p : sched.patch_list.global) {
            interface_field_map[p.id_patch] = std::vector<std::tuple<u64, std::unique_ptr<PCField>>>();
        }

        auto t1 = timings::start_timer("generate interfaces", timings::timingtype::function);
        std::vector<std::unique_ptr<PCField>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if (interface_comm_list.size() > 0) {

            for (u64 i = 0; i < interface_comm_list.size(); i++) {

                if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0) {


                    std::vector<std::unique_ptr<PCField>> pret = InterfaceVolumeGenerator::append_interface_field<T,vectype>(
                        hndl.get_queue_alt(0), sched.patch_data.owned_data[interface_comm_list[i].sender_patch_id],pcomp_field.field_data[interface_comm_list[i].sender_patch_id],
                        {interface_comm_list[i].interf_box_min}, {interface_comm_list[i].interf_box_max});


                    for (auto &pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                } else {
                    comm_pdat.push_back(std::make_unique<PCField>());
                }
                comm_vec.push_back(
                    u64_2{interface_comm_list[i].global_patch_idx_send, interface_comm_list[i].global_patch_idx_recv});
            }

            //std::cout << "\n split \n";
        }
        t1.stop();

        auto t2 = timings::start_timer("patch_data_exchange_object", timings::timingtype::mpi);
        patch_data_field_exchange_object<T>(sched.patch_list.global, comm_pdat,comm_vec,interface_field_map);
        t2.stop();
    }

} // namespace impl