





#pragma once

#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "algs/syclreduction.hpp"
#include <stdexcept>
#include <type_traits>

namespace patchdata {
    namespace sph {

        template<class DataLayout, class htype>
        inline htype get_h_max(sycl::queue & queue, PatchDataBuffer & pdatbuf){

            if(pdatbuf.element_count == 0) return 0;

            using U = typename DataLayout::template U1<htype>::T;

            htype tmp;

            if constexpr (std::is_same<htype, f32>::value){
                tmp = syclalg::get_max<f32, U::nvar, U::ihpart>(queue, pdatbuf.U1_s);
            } else if constexpr (std::is_same<htype, f64>::value){
                tmp = syclalg::get_max<f64, U::nvar, U::ihpart>(queue, pdatbuf.U1_d);
            }else{
                throw shamrock_exc("get_h_max -> current htype not handled");
            }

            return tmp;

        }


        template<class vec>
        inline std::tuple<vec,vec> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf);

        template<>
        inline std::tuple<f32_3,f32_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){

            return syclalg::get_min_max<f32_3, 1,0>(queue, pdatbuf.pos_s);

        }

        template<>
        inline std::tuple<f64_3,f64_3> get_patchdata_BBAA(sycl::queue & queue,PatchDataBuffer & pdatbuf){

            return syclalg::get_min_max<f64_3, 1,0>(queue, pdatbuf.pos_d);

        }


    }
}