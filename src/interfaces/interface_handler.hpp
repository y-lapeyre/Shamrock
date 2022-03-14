#pragma once

#include <memory>
#include <vector>

#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "patchscheduler/scheduler_mpi.hpp"

template <class vectype, class primtype> class InterfaceHandler {

  private:
    std::vector<InterfaceComm<vectype>> interface_comm_list;
    std::unordered_map<u64,std::vector<std::tuple<u64,std::unique_ptr<PatchData>>>> interface_map;

  public:
    template <class interface_selector>
    inline void compute_interface_list(SchedulerMPI &sched, SerialPatchTree<vectype> sptree, PatchField<primtype> h_field) {
        interface_comm_list = Interface_Generator<vectype, primtype, interface_selector>::get_interfaces_comm_list(
            sched, sptree, h_field, format("interfaces_%d_node%d", 0, mpi_handler::world_rank));
    }

    void comm_interfaces(SchedulerMPI &sched){
        SyCLHandler &hndl = SyCLHandler::get_instance();


        interface_map.clear();
        for (const Patch & p : sched.patch_list.global) {
            interface_map[p.id_patch] = std::vector<std::tuple<u64,std::unique_ptr<PatchData>>>();
        }

        
        std::vector<std::unique_ptr<PatchData>> comm_pdat;
        std::vector<u64_2> comm_vec;
        if(interface_comm_list.size() > 0){
            
            

            for (u64 i = 0 ; i < interface_comm_list.size(); i++) {
                
                if(sched.patch_list.global[interface_comm_list[i].global_patch_idx_send].data_count > 0){
                    std::vector<std::unique_ptr<PatchData>> pret = InterfaceVolumeGenerator::append_interface<vectype>( 
                        hndl.alt_queues[0], 
                        sched.patch_data.owned_data[interface_comm_list[i].sender_patch_id], 
                        {interface_comm_list[i].interf_box_min}, 
                        {interface_comm_list[i].interf_box_max});
                    for (auto & pdat : pret) {
                        comm_pdat.push_back(std::move(pdat));
                    }
                }else{
                    comm_pdat.push_back(std::make_unique<PatchData>());
                }
                comm_vec.push_back(u64_2{
                    interface_comm_list[i].global_patch_idx_send,interface_comm_list[i].global_patch_idx_recv
                });

            }

            std::cout << "\n split \n";

        }

        std::cout << "len comm_pdat : " << comm_pdat.size() << std::endl;
        std::cout << "len comm_vec : " << comm_vec.size() << std::endl;
        
        std::vector<i32> local_comm_tag(comm_vec.size());
        {
            i32 iterator = 0; 
            for(u64 i = 0 ; i < comm_vec.size(); i++){
                const Patch & psend = sched.patch_list.global[comm_vec[i].x()];
                const Patch & precv = sched.patch_list.global[comm_vec[i].y()];

                local_comm_tag[i] = iterator;

                iterator++;
            }

            
        }


        
        std::vector<u64_2> global_comm_vec;
        std::vector<i32> global_comm_tag;
        mpi_handler::vector_allgatherv(comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
        mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);

        


        std::vector<MPI_Request> rq_lst;

        {
            for(u64 i = 0 ; i < comm_vec.size(); i++){
                const Patch & psend = sched.patch_list.global[comm_vec[i].x()];
                const Patch & precv = sched.patch_list.global[comm_vec[i].y()];

                if(psend.node_owner_id == precv.node_owner_id){
                    //std::cout << "same node !!!\n";
                    interface_map[precv.id_patch].push_back({psend.id_patch, std::move(comm_pdat[i])});
                    comm_pdat[i] = nullptr;
                }else{
                    std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
                    patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
                }

                // std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
                // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
            }
           
        }




        if(global_comm_vec.size() > 0){


            std::cout << std::endl;
            for(u64 i = 0 ; i < global_comm_vec.size(); i++){


                const Patch & psend = sched.patch_list.global[global_comm_vec[i].x()];
                const Patch & precv = sched.patch_list.global[global_comm_vec[i].y()];
                //std::cout << format("(%3d,%3d) : %d -> %d / %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

                if(precv.node_owner_id == mpi_handler::world_rank){

                    if(psend.node_owner_id != precv.node_owner_id){
                        std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                        interface_map[precv.id_patch].push_back({psend.id_patch, std::make_unique<PatchData>()});//patchdata_irecv(recv_rq, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                        patchdata_irecv(*std::get<1>(interface_map[precv.id_patch][interface_map[precv.id_patch].size()-1]),rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                    }

                    // std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                    // Interface_map[precv.id_patch].push_back({psend.id_patch, new PatchData()});//patchdata_irecv(recv_rq, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                    // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }

            }
            std::cout << std::endl;
            

        }


        std::vector<MPI_Status> st_lst(rq_lst.size());
        mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());
        

    }

    const std::vector<std::tuple<u64,std::unique_ptr<PatchData>>> & get_interface_list(u64 key){
        return interface_map[key];
    }


    inline void print_current_interf_map(){

        for (const auto & [pid,int_vec] : interface_map) {
            printf(" pid : %d :\n", pid);
            for(auto & [a,b] : int_vec){
                printf("    -> %d : len %d\n",a,b->pos_s.size() + b->pos_d.size());
            }
        }

    }



};