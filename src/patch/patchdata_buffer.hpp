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
        sycl::buffer<f32_3> (pdat.pos_s),
        sycl::buffer<f64_3> (pdat.pos_d),
        sycl::buffer<f32>   (pdat.U1_s ),
        sycl::buffer<f64>   (pdat.U1_d ),
        sycl::buffer<f32_3> (pdat.U3_s ),
        sycl::buffer<f64_3> (pdat.U3_d )    
    };
}
