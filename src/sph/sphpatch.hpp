





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

            using U = typename DataLayout::template U1<htype>;

            htype tmp;

            if constexpr (std::is_same<htype, f32>::value){
                tmp = syclalg::get_max<f32, U::nvar, U::ihpart>(queue, pdatbuf.U1_s);
            } else if constexpr (std::is_same<htype, f64>::value){
                tmp = syclalg::get_max<f64, U::nvar, U::ihpart>(queue, pdatbuf.U1_d);
            }else{
                throw std::runtime_error("get_h_max -> current htype not handled");
            }

            return tmp;

        }


    }
}