/**
 * @file patchdata_buffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <stdexcept>
#include <unordered_map>

#include "patch/patchdata.hpp"
#include "sys/sycl_handler.hpp"

/**
 * @brief sycl buffer loaded version of PatchData
 * 
 */
class PatchDataBuffer{ public:

    u32 element_count;


    sycl::buffer<f32_3> pos_s; ///< f32 's for position         
    sycl::buffer<f64_3> pos_d; ///< f64 's for position         
    sycl::buffer<f32>   U1_s ; ///< f32 's for internal fields  
    sycl::buffer<f64>   U1_d ; ///< f64 's for internal fields  
    sycl::buffer<f32_3> U3_s ; ///< f32_3 's for internal fields
    sycl::buffer<f64_3> U3_d ; ///< f64_3 's for internal fields
};


inline PatchDataBuffer attach_to_patchData(PatchData & pdat){
    return PatchDataBuffer{
        u32(pdat.pos_s.size() + pdat.pos_d.size()),
        sycl::buffer<f32_3> (pdat.pos_s.data(),pdat.pos_s.size()),
        sycl::buffer<f64_3> (pdat.pos_d.data(),pdat.pos_d.size()),
        sycl::buffer<f32>   (pdat.U1_s .data(),pdat.U1_s .size()),
        sycl::buffer<f64>   (pdat.U1_d .data(),pdat.U1_d .size()),
        sycl::buffer<f32_3> (pdat.U3_s .data(),pdat.U3_s .size()),
        sycl::buffer<f64_3> (pdat.U3_d .data(),pdat.U3_d .size())    
    };
}
