#include "sys/mpi_handler.hpp"

#include "patchdata_exchanger.hpp"

void patch_data_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchData>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> & recv_obj
    ){

    //TODO enable if ultra verbose
    // std::cout << "len comm_pdat : " << send_comm_pdat.size() << std::endl;
    // std::cout << "len comm_vec : " << send_comm_vec.size() << std::endl;

    std::vector<i32> local_comm_tag(send_comm_vec.size());
    {
        i32 iterator = 0;
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            local_comm_tag[i] = iterator;

            iterator++;
        }
    }

    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;
    mpi_handler::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
    mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);

    std::vector<MPI_Request> rq_lst;

    {
        for (u64 i = 0; i < send_comm_vec.size(); i++) {
            const Patch &psend = global_patch_list[send_comm_vec[i].x()];
            const Patch &precv = global_patch_list[send_comm_vec[i].y()];

            if (psend.node_owner_id == precv.node_owner_id) {
                // std::cout << "same node !!!\n";
                recv_obj[precv.id_patch].push_back({psend.id_patch, std::move(send_comm_pdat[i])});
                send_comm_pdat[i] = nullptr;
            } else {
                //TODO enable if ultra verbose
                //std::cout << format("send : (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                //                    psend.node_owner_id, precv.node_owner_id, local_comm_tag[i]);
                patchdata_isend(*send_comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
            }

            // std::cout << format("send : (%3d,%3d) : %d -> %d /
            // %d\n",psend.id_patch,precv.id_patch,psend.node_owner_id,precv.node_owner_id,local_comm_tag[i]);
            // patchdata_isend(* comm_pdat[i], rq_lst, precv.node_owner_id, local_comm_tag[i], MPI_COMM_WORLD);
        }
    }

    if (global_comm_vec.size() > 0) {

        std::cout << std::endl;
        for (u64 i = 0; i < global_comm_vec.size(); i++) {

            const Patch &psend = global_patch_list[global_comm_vec[i].x()];
            const Patch &precv = global_patch_list[global_comm_vec[i].y()];
            // std::cout << format("(%3d,%3d) : %d -> %d /
            // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,iterator);

            if (precv.node_owner_id == mpi_handler::world_rank) {

                if (psend.node_owner_id != precv.node_owner_id) {
                    //TODO enable if ultra verbose
                    // std::cout << format("recv (%3d,%3d) : %d -> %d / %d\n", psend.id_patch, precv.id_patch,
                    //                     psend.node_owner_id, precv.node_owner_id, global_comm_tag[i]);
                    recv_obj[precv.id_patch].push_back(
                        {psend.id_patch, std::make_unique<PatchData>()}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                                                          // global_comm_tag[i], MPI_COMM_WORLD)}
                    patchdata_irecv(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                    rq_lst, psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
                }

                // std::cout << format("recv (%3d,%3d) : %d -> %d /
                // %d\n",global_comm_vec[i].x(),global_comm_vec[i].y(),psend.node_owner_id,precv.node_owner_id,global_comm_tag[i]);
                // Interface_map[precv.id_patch].push_back({psend.id_patch, new PatchData()});//patchdata_irecv(recv_rq,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD)}
                // patchdata_irecv(*std::get<1>(Interface_map[precv.id_patch][Interface_map[precv.id_patch].size()-1]),rq_lst,
                // psend.node_owner_id, global_comm_tag[i], MPI_COMM_WORLD);
            }
        }
        std::cout << std::endl;
    }

    std::vector<MPI_Status> st_lst(rq_lst.size());
    mpi::waitall(rq_lst.size(), rq_lst.data(), st_lst.data());

    
    //TODO check that this sort is valid
    for(auto & [key,obj] : recv_obj){
        std::sort(obj.begin(), obj.end());
    }

}
