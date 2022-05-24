// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//







#pragma once

#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "algs/syclreduction.hpp"
#include "patch/patchdata_layout.hpp"
#include <stdexcept>
#include <type_traits>

#include "sph_aliases.hpp"

namespace patchdata {
    namespace sph {

        template<class htype>
        inline htype get_h_max(PatchDataLayout & pdl,sycl::queue & queue, PatchDataBuffer & pdatbuf){

            if(pdatbuf.element_count == 0) return 0;

            htype tmp;

            if constexpr (std::is_same<htype, f32>::value){

                u32 ihpart = pdl.get_field_idx<f32>(::sph::field_names::field_hpart);
                tmp = syclalg::get_max<f32>(queue, pdatbuf.fields_f32[ihpart]);

            } else if constexpr (std::is_same<htype, f64>::value){
                u32 ihpart = pdl.get_field_idx<f64>(::sph::field_names::field_hpart);
                tmp = syclalg::get_max<f64>(queue, pdatbuf.fields_f64[ihpart]);
                
            }else{
                throw shamrock_exc("get_h_max -> current htype not handled");
            }

            return tmp;

        }


        template<class vec>
        inline std::tuple<vec,vec> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf);

        template<>
        inline std::tuple<f32_3,f32_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){

            u32 ihpart = pdatbuf.pdl.get_field_idx<f32_3>("xyz");
            return syclalg::get_min_max<f32_3>(queue, pdatbuf.fields_f32_3[ihpart]);

        }

        template<>
        inline std::tuple<f64_3,f64_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){
            u32 ihpart = pdatbuf.pdl.get_field_idx<f64_3>("xyz");
            return syclalg::get_min_max<f64_3>(queue, pdatbuf.fields_f64_3[ihpart]);

        }


    }
}