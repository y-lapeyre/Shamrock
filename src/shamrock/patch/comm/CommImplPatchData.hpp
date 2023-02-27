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
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shamrock/patch/comm/CommImplPatchDataField.hpp"
#include "shamutils/throwUtils.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<>
    class CommDetails<shamrock::patch::PatchData> {

        public:
        u32 obj_cnt;
        std::optional<u64> start_index;
        
        shamrock::patch::PatchDataLayout & pdl;

        inline CommDetails (shamrock::patch::PatchData & pdat) : pdl(pdat.pdl) , obj_cnt(pdat.get_obj_cnt()), start_index({}){}

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
        static var_t to_variant(CommBuffer<PatchDataField<T>, comm_mode> && base){
            return var_t{std::move(base)};
        }



        std::vector<var_t> fields_bufs;


        public:
        inline CommBuffer(CommDetails<PatchData> det)
            : details(det) {

            details.pdl.for_each_field_any([&](auto & field){
                using f_t = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                fields_bufs.push_back(
                    PatchDataFieldCommBuf_t<f_t>{PatchDataFieldCommDet_t<f_t>{
                        details.obj_cnt,
                        field.name,
                        field.nvar,
                        details.start_index
                        }}
                );

            });

        }

        inline CommBuffer(PatchData &pdat) : details(pdat){

            pdat.for_each_field_any([&](auto & field){
                
                using T = typename std::remove_reference<decltype(field)>::type::Field_type;

                PatchDataFieldCommBuf_t<T> comm_buf_field{field};

                var_t variant_comm_buf_field = to_variant(std::move(comm_buf_field));

                fields_bufs.push_back(
                    std::move(variant_comm_buf_field)
                );
                
            });

        }

        inline CommBuffer(PatchData &pdat, CommDetails<PatchData> det) : details(det){

            details.pdl.for_each_field_any([&](auto & field){
                using f_t = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                u32 idx = pdat.pdl.get_field_idx<f_t>(field.name, field.nvar);

                PatchDataField<f_t> & field_recov = pdat.get_field<f_t>(idx);

                fields_bufs.push_back(
                    PatchDataFieldCommBuf_t<f_t>{
                        field_recov, 
                        CommDetails<f_t>(
                            details.obj_cnt, field.name, field.nvar, details.start_index
                        )
                    }
                );

            });

        }



        inline CommBuffer(PatchData &&pdat) : details(pdat){

            pdat.for_each_field_any([&](auto & field){
                
                using T = typename std::remove_reference<decltype(field)>::type::Field_type;

                fields_bufs.push_back(
                    PatchDataFieldCommBuf_t<T>{field}
                );
                
            });

        }

        inline CommBuffer(PatchData &&pdat, CommDetails<PatchData> det) : details(det){

            details.pdl.for_each_field_any([&](auto & field){
                using f_t = typename std::remove_reference<decltype(field)>::type;
                using base_t = typename f_t::field_T;

                u32 idx = pdat.pdl.get_field_idx<f_t>(field.name, field.nvar);

                PatchDataField<f_t> & field_recov = pdat.get_field<f_t>(idx);

                fields_bufs.push_back(
                    PatchDataFieldCommBuf_t<f_t>{
                        field_recov, 
                        CommDetails<f_t>(
                            details.obj_cnt, field.name, field.nvar, details.start_index
                        )
                    }
                );

            });

        }

        




    };

} // namespace shamsys::comm::details