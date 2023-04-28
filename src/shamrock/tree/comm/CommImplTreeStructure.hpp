// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "shamrock/tree/TreeStructure.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shambase/exception.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<class u_morton>
    class CommDetails<shamrock::tree::TreeStructure<u_morton>> {

        public:
        u32 internal_cell_count;
        bool one_cell_mode;

        inline CommDetails(
            u32 internal_cell_count, bool one_cell_mode
        )
            : internal_cell_count(internal_cell_count), one_cell_mode(one_cell_mode){}

        inline CommDetails(shamrock::tree::TreeStructure<u_morton> &strc)
            : internal_cell_count(strc.internal_cell_count), one_cell_mode(strc.one_cell_mode) {}
    };


    template<class u_morton, Protocol comm_mode>
    class CommBuffer<shamrock::tree::TreeStructure<u_morton>, comm_mode> {

        using TreeStructure = shamrock::tree::TreeStructure<u_morton>;
        using CDetails = CommDetails<shamrock::tree::TreeStructure<u_morton>>;

        CDetails details;

        CommBuffer<sycl::buffer<u32>,comm_mode> buf_lchild_id;  // size = internal
        CommBuffer<sycl::buffer<u32>,comm_mode> buf_rchild_id;  // size = internal
        CommBuffer<sycl::buffer<u8>,comm_mode> buf_lchild_flag; // size = internal
        CommBuffer<sycl::buffer<u8>,comm_mode> buf_rchild_flag; // size = internal
        CommBuffer<sycl::buffer<u32>,comm_mode> buf_endrange;   // size = internal (+1 if one cell mode)

        CommBuffer(
            
            CommBuffer<sycl::buffer<u32>,comm_mode>&& buf_lchild_id,  // size = internal
            CommBuffer<sycl::buffer<u32>,comm_mode> && buf_rchild_id,  // size = internal
            CommBuffer<sycl::buffer<u8>,comm_mode> && buf_lchild_flag, // size = internal
            CommBuffer<sycl::buffer<u8>,comm_mode> && buf_rchild_flag, // size = internal
            CommBuffer<sycl::buffer<u32>,comm_mode> && buf_endrange,
            CDetails && details
        )
            : 
            buf_lchild_id(std::move(buf_lchild_id)), 
            buf_rchild_id(std::move(buf_rchild_id)), 
            buf_lchild_flag(std::move(buf_lchild_flag)), 
            buf_rchild_flag(std::move(buf_rchild_flag)), 
            buf_endrange(std::move(buf_endrange)), 
            details(std::move(details)) {}

        public:

        inline CommBuffer(CDetails det) :
            details(det),
            buf_lchild_id{CommDetails<sycl::buffer<u32>>{}},
            buf_rchild_id{CommDetails<sycl::buffer<u32>>{}},
            buf_lchild_flag{CommDetails<sycl::buffer<u8>>{}},
            buf_rchild_flag{CommDetails<sycl::buffer<u8>>{}},
            buf_endrange{CommDetails<sycl::buffer<u32>>{}}
        {}

        inline CommBuffer(TreeStructure &obj_ref)
            : details(obj_ref),
              buf_lchild_id(*obj_ref.buf_lchild_id) ,
              buf_rchild_id(*obj_ref.buf_rchild_id) ,
              buf_lchild_flag(*obj_ref.buf_lchild_flag) ,
              buf_rchild_flag(*obj_ref.buf_rchild_flag) ,
              buf_endrange(*obj_ref.buf_endrange)
              {}

        /*
        
        public:

        inline CommBuffer(PatchDataField<T> &obj_ref)
            : details(obj_ref),
              buf_comm(
                  *obj_ref.get_buf(), CommDetails<PatchDataField<T>>{obj_ref}._get_buf_details()
              ) {}

        inline CommBuffer(PatchDataField<T> &obj_ref, CommDetails<PatchDataField<T>> det)
            : details(obj_ref), buf_comm(*obj_ref.get_buf(), det._get_buf_details()) {}

        inline CommBuffer(PatchDataField<T> &&moved_obj)
            : details(moved_obj), buf_comm(
                                      PatchDataField<T>::convert_to_buf(std::move(moved_obj)),
                                      CommDetails<PatchDataField<T>>{moved_obj}._get_buf_details()
                                  ) {}

        inline CommBuffer(PatchDataField<T> &&moved_obj, CommDetails<PatchDataField<T>> det)
            : details(moved_obj),
              buf_comm(
                  PatchDataField<T>::convert_to_buf(std::move(moved_obj)), det._get_buf_details()
              ) {}

        inline PatchDataField<T> copy_back() {
            sycl::buffer<T> buf = buf_comm.copy_back();

            return PatchDataField<T>{std::move(buf), details.obj_cnt, details.name, details.nvar};
        }
        // void copy_back(PatchDataField<T> & dest);
        inline static PatchDataField<T> convert(CommBuffer &&buf) {
            sycl::buffer<T> buf_recov = buf.buf_comm.copy_back();

            return PatchDataField<T>{
                std::move(buf_recov), buf.details.obj_cnt, buf.details.name, buf.details.nvar};
        }

        inline void isend(CommRequests &rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm) {
            buf_comm.isend(rqs, rank_dest, comm_flag, comm);
        }
        inline void irecv(CommRequests &rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm) {
            buf_comm.irecv(rqs, rank_src, comm_flag, comm);
        }

        inline static CommBuffer irecv_probe(
            CommRequests &rqs,
            u32 rank_src,
            u32 comm_flag,
            MPI_Comm comm,
            CommDetails<PatchDataField<T>> details
        ) {

            auto recv = CommBuffer<sycl::buffer<T>, comm_mode>::irecv_probe(
                rqs, rank_src, comm_flag, comm, {}
            );

            u64 cnt_recv = recv.get_details().comm_len;

            if (cnt_recv % details.nvar != 0) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "the received message must be disible by nvar to be received as PatchDataField"
                );
            }

            details.obj_cnt = cnt_recv / details.nvar;

            return CommBuffer{std::move(recv), std::move(details)};
        }

        */

    };

}