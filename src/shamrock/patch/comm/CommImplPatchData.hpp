// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "shamrock/patch/PatchData.hpp"
#include "shamrock/patch/comm/CommImplPatchDataField.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shambase/throwUtils.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<>
    class CommDetails<shamrock::patch::PatchData> {

        public:
        u32 obj_cnt;
        std::optional<u64> start_index;

        shamrock::patch::PatchDataLayout &pdl;

        inline CommDetails(
            u32 obj_cnt, std::optional<u64> start_index, shamrock::patch::PatchDataLayout &pdl
        )
            : obj_cnt(obj_cnt), start_index(start_index), pdl(pdl) {}

        inline CommDetails(shamrock::patch::PatchData &pdat)
            : pdl(pdat.pdl), obj_cnt(pdat.get_obj_cnt()), start_index({}) {}
    };

    template<class T>
    using PatchDataFieldCommDet_t = CommDetails<PatchDataField<T>>;

    template<Protocol comm_mode>
    class CommBuffer<shamrock::patch::PatchData, comm_mode> {

        using PatchData = shamrock::patch::PatchData;

        CommDetails<PatchData> details;

        template<class T>
        using PatchDataFieldCommDet_t = CommDetails<PatchDataField<T>>;

        template<class T>
        using PatchDataFieldCommBuf_t = CommBuffer<PatchDataField<T>, comm_mode>;

        using var_t = shamrock::patch::FieldVariant<PatchDataFieldCommBuf_t>;

        template<class T>
        static var_t to_variant(CommBuffer<PatchDataField<T>, comm_mode> &&base) {
            return var_t{std::move(base)};
        }

        std::vector<var_t> fields_bufs;

        CommBuffer(std::vector<var_t> &&fields_bufs, CommDetails<PatchData> &&details)
            : fields_bufs(std::move(fields_bufs)), details(details) {}

        public:
        inline CommBuffer(CommDetails<PatchData> det) : details(det) {

            details.pdl.for_each_field_any([&](auto &field) {
                using f_t    = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                fields_bufs.push_back(var_t{PatchDataFieldCommBuf_t<base_t>{PatchDataFieldCommDet_t<base_t>{
                    details.obj_cnt, field.name, field.nvar, details.start_index}}});
            });
        }

        inline CommBuffer(PatchData &pdat) : details(pdat) {

            pdat.for_each_field_any([&](auto &field) {
                using T = typename std::remove_reference<decltype(field)>::type::Field_type;

                PatchDataFieldCommBuf_t<T> comm_buf_field{field};

                var_t variant_comm_buf_field = to_variant(std::move(comm_buf_field));

                fields_bufs.push_back(std::move(variant_comm_buf_field));
            });
        }

        inline CommBuffer(PatchData &pdat, CommDetails<PatchData> det) : details(det) {

            details.pdl.for_each_field_any([&](auto &field) {
                using f_t    = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                u32 idx = pdat.pdl.get_field_idx<f_t>(field.name, field.nvar);

                PatchDataField<base_t> &field_recov = pdat.get_field<base_t>(idx);

                fields_bufs.push_back(PatchDataFieldCommBuf_t<base_t>{
                    field_recov,
                    CommDetails<base_t>(details.obj_cnt, field.name, field.nvar, details.start_index)}
                );
            });
        }

        inline CommBuffer(PatchData &&pdat) : details(pdat) {

            pdat.for_each_field_any([&](auto &field) {
                using T = typename std::remove_reference<decltype(field)>::type::Field_type;

                fields_bufs.push_back(PatchDataFieldCommBuf_t<T>{field});
            });
        }

        inline CommBuffer(PatchData &&pdat, CommDetails<PatchData> det) : details(det) {

            details.pdl.for_each_field_any([&](auto &field) {
                using f_t    = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                u32 idx = pdat.pdl.get_field_idx<base_t>(field.name, field.nvar);

                PatchDataField<base_t> &field_recov = pdat.get_field<base_t>(idx);

                fields_bufs.push_back(PatchDataFieldCommBuf_t<base_t>{
                    field_recov,
                    CommDetails<base_t>(details.obj_cnt, field.name, field.nvar, details.start_index)}
                );
            });
        }

        inline PatchData copy_back() {

            return PatchData{
                details.pdl, [&](auto &pdat_fields) {
                    for (var_t &recov : fields_bufs) {
                        recov.visit([&](auto &buf) {
                            pdat_fields.push_back(PatchData::field_variant_t{buf.copy_back()});
                        });
                    }
                }};
        }

        inline static PatchData convert(CommBuffer &&buf) {

            return PatchData{
                buf.details.pdl, [&](auto &pdat_fields) {
                    for (var_t &recov : buf.fields_bufs) {
                        recov.visit([&](auto &buf) {
                            pdat_fields.push_back(PatchData::field_variant_t{buf.copy_back()});
                        });
                    }
                }};
        }

        inline void isend(CommRequests &rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm) {
            for (var_t &recov : fields_bufs) {
                recov.visit([&](auto &buf_comm) { buf_comm.isend(rqs, rank_dest, comm_flag, comm); }
                );
            }
        }
        inline void irecv(CommRequests &rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm) {
            for (var_t &recov : fields_bufs) {
                recov.visit([&](auto &buf_comm) { buf_comm.irecv(rqs, rank_src, comm_flag, comm); }
                );
            }
        }

        inline static CommBuffer irecv_probe(
            CommRequests &rqs,
            u32 rank_src,
            u32 comm_flag,
            MPI_Comm comm,
            CommDetails<PatchData> details
        ) {

            std::vector<var_t> fields_bufs;

            details.pdl.for_each_field_any([&](auto &field) {
                using f_t    = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                PatchDataFieldCommBuf_t<base_t> recv_buf = PatchDataFieldCommBuf_t<base_t>::irecv_probe(
                    rqs,
                    rank_src,
                    comm_flag,
                    comm,
                    PatchDataFieldCommDet_t<base_t>{
                        0, // because the obj count will be overwritten by the probe call
                        field.name,
                        field.nvar,
                        details.start_index}
                );

                var_t var_recv {std::move(recv_buf)};

                fields_bufs.push_back(std::move(var_recv));
            });

            return CommBuffer{std::move(fields_bufs), std::move(details)};
        }
    };

} // namespace shamsys::comm::details