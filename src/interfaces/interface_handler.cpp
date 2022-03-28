#include "interface_handler.hpp"
#include "aliases.hpp"

#include "patch/patchdata_exchanger.hpp"

template <class vectype, class primtype>
void _comm_interfaces(SchedulerMPI &sched, std::vector<InterfaceComm<vectype>> &interface_comm_list,
                      std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> &interface_map) {
    SyCLHandler &hndl = SyCLHandler::get_instance();

    interface_map.clear();
    for (const Patch &p : sched.patch_list.global) {
        interface_map[p.id_patch] = std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>();
    }

    std::vector<std::unique_ptr<PatchData>> comm_pdat;
    std::vector<u64_2> comm_vec;
    if (interface_comm_list.size() > 0) {

        for (u64 i = 0; i < interface_comm_list.size(); i++) {

            if (sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0) {
                std::vector<std::unique_ptr<PatchData>> pret = InterfaceVolumeGenerator::append_interface<vectype>(
                    hndl.get_queue_alt(0), sched.patch_data.owned_data[interface_comm_list[i].sender_patch_id],
                    {interface_comm_list[i].interf_box_min}, {interface_comm_list[i].interf_box_max});
                for (auto &pdat : pret) {
                    comm_pdat.push_back(std::move(pdat));
                }
            } else {
                comm_pdat.push_back(std::make_unique<PatchData>());
            }
            comm_vec.push_back(
                u64_2{interface_comm_list[i].global_patch_idx_send, interface_comm_list[i].global_patch_idx_recv});
        }

        std::cout << "\n split \n";
    }

    patch_data_exchange_object(sched.patch_list.global, comm_pdat,comm_vec,interface_map);
}

template <> void InterfaceHandler<f32_3, f32>::comm_interfaces(SchedulerMPI &sched) {
    _comm_interfaces<f32_3, f32>(sched, interface_comm_list, interface_map);
}

template <> void InterfaceHandler<f64_3, f64>::comm_interfaces(SchedulerMPI &sched) {
    _comm_interfaces<f64_3, f64>(sched, interface_comm_list, interface_map);
}