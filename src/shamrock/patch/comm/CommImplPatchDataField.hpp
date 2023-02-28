// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"
#include "shamutils/throwUtils.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<class T>
    class CommDetails<PatchDataField<T>> {
        public:
        u32 obj_cnt;
        std::string name;
        u32 nvar;
        std::optional<u64> start_index;

        CommDetails(u32 obj_cnt, std::string name, u32 nvar, std::optional<u64> start_index)
            : obj_cnt(obj_cnt), name(name), nvar(nvar), start_index(start_index) {}

        CommDetails(PatchDataField<T> &pdat)
            : obj_cnt(pdat.get_obj_cnt()), name(pdat.get_name()), nvar(pdat.get_nvar()),
              start_index({}) {}

        CommDetails(PatchDataField<T> &pdat, u32 start_index)
            : obj_cnt(pdat.get_obj_cnt()), name(pdat.get_name()), nvar(pdat.get_nvar()),
              start_index(pdat.get_nvar() * start_index) {}

        CommDetails<sycl::buffer<T>> _get_buf_details() {
            if (start_index.has_value()) {
                return CommDetails<sycl::buffer<T>>{obj_cnt * nvar, (*start_index) * nvar};
            }

            return CommDetails<sycl::buffer<T>>{obj_cnt * nvar, {}};
        }
    };

    template<class T, Protocol comm_mode>
    class CommBuffer<PatchDataField<T>, comm_mode> {

        CommBuffer<sycl::buffer<T>, comm_mode> buf_comm;
        CommDetails<PatchDataField<T>> details;

        CommBuffer(
            CommBuffer<sycl::buffer<T>, comm_mode> &&buf_comm,
            CommDetails<PatchDataField<T>> &&details
        )
            : buf_comm(std::move(buf_comm)), details(std::move(details)) {}

        public:
        inline CommBuffer(CommDetails<PatchDataField<T>> det)
            : details(det), buf_comm{det._get_buf_details()} {}

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
                throw shamutils::throw_with_loc<std::runtime_error>(
                    "the received message must be disible by nvar to be received as PatchDataField"
                );
            }

            details.obj_cnt = cnt_recv / details.nvar;

            return CommBuffer{std::move(recv), std::move(details)};
        }
    };

} // namespace shamsys::comm::details