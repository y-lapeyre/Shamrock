// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "CommBuffer.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shambase/stacktrace.hpp"
#include "shamsys/SyclMpiTypes.hpp"

namespace shamsys::comm {

    template<class T>
    using SparseCommSource = std::vector<std::unique_ptr<T>>;

    template<class T>
    using SparseCommResult =
        std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<T>>>>;

    template<class T>
    using SparseCommSourceBuffers = SparseCommSource<CommBuffer<T>>;

    template<class T>
    using SparseCommResultBuffers = SparseCommResult<CommBuffer<T>>;

    /**
     * @brief N object distributed on p processes, the comm matrix
     * is defined on the N objects and not p processes
     */
    class SparseGroupCommunicator {

        public:
        std::vector<u64_2> global_comm_vec;
        std::vector<i32> global_comm_tag;

        std::vector<u64_2> local_comm_vec;
        std::vector<i32> local_comm_tag;

        SparseGroupCommunicator(std::vector<u64_2> local_comm_vec)
            : local_comm_vec(local_comm_vec), local_comm_tag(local_comm_vec.size()) {
            StackEntry stack_loc{};

            {
                i32 iterator = 0;
                for (u64 i = 0; i < local_comm_vec.size(); i++) {
                    local_comm_tag[i] = iterator;
                    iterator++;
                }
            }

            shamalgs::collective::vector_allgatherv(local_comm_vec,
                                           get_mpi_type<u64_2>(),
                                           global_comm_vec,
                                           get_mpi_type<u64_2>(),
                                           MPI_COMM_WORLD);
            shamalgs::collective::vector_allgatherv(
                local_comm_tag, get_mpi_type<i32>(), global_comm_tag, get_mpi_type<i32>(), MPI_COMM_WORLD);
        }

        template<class T, class Func, class Func2>
        inline SparseCommResultBuffers<T>
        sparse_exchange(const SparseCommSourceBuffers<T> &send_objs,
                        CommDetails<T> details,
                        Protocol comm_mode,
                        Func &&rank_id_getter,
                        Func2 &&id_setter) {

            StackEntry stack_loc{};

            SparseCommResultBuffers<T> recv_obj;

            if (!send_objs.empty()) {

                CommRequests rqs;

                {
                    for (u64 i = 0; i < local_comm_vec.size(); i++) {

                        const u32 rank_send = rank_id_getter(local_comm_vec[i].x());
                        const u32 rank_recv = rank_id_getter(local_comm_vec[i].y());

                        if (rank_send == rank_recv) {

                            u64 id_obj_send = id_setter(local_comm_vec[i].x());
                            u64 id_obj_recv = id_setter(local_comm_vec[i].y());

                            auto &vec = recv_obj[id_obj_recv];

                            vec.push_back({id_obj_send, send_objs[i]->duplicate_to_ptr()});
                        } else {
                            send_objs[i]->isend(rqs, rank_recv, local_comm_tag[i], MPI_COMM_WORLD);
                        }
                    }
                }

                if (global_comm_vec.size() > 0) {

                    for (u64 i = 0; i < global_comm_vec.size(); i++) {

                        const u32 rank_send = rank_id_getter(global_comm_vec[i].x());
                        const u32 rank_recv = rank_id_getter(global_comm_vec[i].y());

                        u64 id_obj_send = id_setter(global_comm_vec[i].x());
                        u64 id_obj_recv = id_setter(global_comm_vec[i].y());

                        if (rank_recv == shamsys::instance::world_rank) {

                            if (rank_send != rank_recv) {

                                CommBuffer<T> recv_buf =
                                    CommBuffer<T>::irecv_probe(rqs,
                                                               rank_send,
                                                               global_comm_tag[i],
                                                               MPI_COMM_WORLD,
                                                               comm_mode,
                                                               details);

                                recv_obj[id_obj_recv].push_back(
                                    {id_obj_send,
                                     std::make_unique<CommBuffer<T>>(std::move(recv_buf))});
                            }
                        }
                    }
                }

                rqs.wait_all();

                for (auto &[key, obj] : recv_obj) {
                    std::sort(obj.begin(), obj.end(), [](const auto &lhs, const auto &rhs) {
                        return std::get<0>(lhs) < std::get<0>(rhs);
                    });
                }
            }

            return recv_obj;
        }
    };

} // namespace shamsys::comm