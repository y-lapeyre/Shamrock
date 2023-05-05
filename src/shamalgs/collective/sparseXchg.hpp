// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/memory/serialize.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommunicationBuffer.hpp"
#include "shambase/integer.hpp"

namespace shamalgs::collective {

    struct SparseCommTable {
        std::vector<u64_2> send_vec_comm_index;
        std::vector<i32> send_vec_comm_tag;

        std::vector<u64_2> global_comm_index;
        std::vector<i32> global_comm_tag;

        inline void build_send_vec_tags() {
            StackEntry stack_loc{};
            send_vec_comm_tag.resize(send_vec_comm_index.size());
            i32 iterator = 0;
            for (u64 i = 0; i < send_vec_comm_index.size(); i++) {
                send_vec_comm_tag[i] = iterator;
                iterator++;
            }
        }
    };

    inline SparseCommTable build_comm_table(std::vector<u64_2> &&send_vec_comm_index) {

        StackEntry stack_loc{};

        SparseCommTable comm_table;

        comm_table.send_vec_comm_index = std::forward<std::vector<u64_2>>(send_vec_comm_index);

        comm_table.build_send_vec_tags();

        vector_allgatherv(
            comm_table.send_vec_comm_index, comm_table.global_comm_index, MPI_COMM_WORLD);
        vector_allgatherv(comm_table.send_vec_comm_tag, comm_table.global_comm_tag, MPI_COMM_WORLD);

        return comm_table;
    }

    struct SendPayload{
        i32 receiver_rank;
        std::unique_ptr<shamsys::CommunicationBuffer> payload;
    };

    struct RecvPayload{
        i32 sender_ranks;
        std::unique_ptr<shamsys::CommunicationBuffer> payload;
    };

    template<class Func>
    inline void base_sparse_comm(
        const std::vector<SendPayload> & message_send,
        std::vector<RecvPayload> & message_recv,
        shamsys::CommunicationProtocol protocol
        ) 
    {
        using namespace shamsys::instance;


        //share comm list accros nodes
        std::vector<u64> send_vec_comm_ranks;

        i32 iterator = 0;
        for (u64 i = 0; i < message_send.size(); i++) {
            send_vec_comm_ranks[i] = shambase::pack(world_rank, message_send[i].receiver_rank);
        }

        std::vector<u64> global_comm_ranks;
        vector_allgatherv(send_vec_comm_ranks, global_comm_ranks, MPI_COMM_WORLD);


        //note the tag cannot be bigger than max_i32 because of the allgatherv

        std::vector<MPI_Request> rqs;

        //send step
        u32 send_idx = 0;
        for(u32 i = 0; i < global_comm_ranks.size(); i++){
            u32_2 comm_ranks = shambase::unpack(global_comm_ranks[i]);

            if(comm_ranks.x() == world_rank){

                auto & payload = message_send[send_idx].payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto & rq = rqs[rq_index]; 

                mpi::isend(
                    payload->get_ptr(), 
                    payload->get_bytesize(), 
                    MPI_BYTE, 
                    comm_ranks.y(), 
                    i, 
                    MPI_COMM_WORLD, 
                    &rq);

                send_idx++;
            }
        }


        //recv step
        for(u32 i = 0; i < global_comm_ranks.size(); i++){
            u32_2 comm_ranks = shambase::unpack(global_comm_ranks[i]);

            if(comm_ranks.y() == world_rank){

                RecvPayload payload;

                rqs.push_back(MPI_Request{});
                u32 rq_index = rqs.size() - 1;
                auto & rq = rqs[rq_index];  

                MPI_Status st;
                i32 cnt;
                mpi::probe(comm_ranks.x(), i,MPI_COMM_WORLD, & st);
                mpi::get_count(&st, get_mpi_type<u32>(), &cnt);

                payload.payload = std::make_unique<shamsys::CommunicationBuffer>(cnt, protocol);

                mpi::irecv(
                    payload.payload->get_ptr(), 
                    cnt, 
                    MPI_BYTE, 
                    comm_ranks.x(), 
                    i, 
                    MPI_COMM_WORLD, 
                    &rq);

                message_recv.push_back(std::move(payload));

            }
        }

        std::vector<MPI_Status> st_lst(rqs.size());
        mpi::waitall(rqs.size(), rqs.data(), st_lst.data());
    }

} // namespace shamalgs::collective