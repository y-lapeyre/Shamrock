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

#include <memory>
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


    std::unique_ptr<sycl::buffer<f32_3>> pos_s; ///< f32 's for position         
    std::unique_ptr<sycl::buffer<f64_3>> pos_d; ///< f64 's for position         
    std::unique_ptr<sycl::buffer<f32>  > U1_s ; ///< f32 's for internal fields  
    std::unique_ptr<sycl::buffer<f64>  > U1_d ; ///< f64 's for internal fields  
    std::unique_ptr<sycl::buffer<f32_3>> U3_s ; ///< f32_3 's for internal fields
    std::unique_ptr<sycl::buffer<f64_3>> U3_d ; ///< f64_3 's for internal fields


    template<class type> std::unique_ptr<sycl::buffer<type>> & get_pos();
    template<> std::unique_ptr<sycl::buffer<f32_3>> & get_pos<f32_3>(){return pos_s;}
    template<> std::unique_ptr<sycl::buffer<f64_3>> & get_pos<f64_3>(){return pos_d;}

    template<class type> std::unique_ptr<sycl::buffer<type>> & get_U1();
    template<> std::unique_ptr<sycl::buffer<f32>> & get_U1<f32>(){return U1_s;}
    template<> std::unique_ptr<sycl::buffer<f64>> & get_U1<f64>(){return U1_d;}

    template<class type> std::unique_ptr<sycl::buffer<type>> & get_U3();
    template<> std::unique_ptr<sycl::buffer<f32_3>> & get_U3<f32_3>(){return U3_s;}
    template<> std::unique_ptr<sycl::buffer<f64_3>> & get_U3<f64_3>(){return U3_d;}

};


inline PatchDataBuffer attach_to_patchData(PatchData & pdat){
    PatchDataBuffer pdatbuf;
    
    pdatbuf.element_count = u32(pdat.pos_s.size() + pdat.pos_d.size());

    if(! pdat.pos_s.empty()) pdatbuf.pos_s = std::make_unique<sycl::buffer<f32_3>>(pdat.pos_s.data(),pdat.pos_s.size());
    if(! pdat.pos_d.empty()) pdatbuf.pos_d = std::make_unique<sycl::buffer<f64_3>>(pdat.pos_d.data(),pdat.pos_d.size());
    if(! pdat.U1_s .empty()) pdatbuf.U1_s  = std::make_unique<sycl::buffer<f32>  >(pdat.U1_s .data(),pdat.U1_s .size());
    if(! pdat.U1_d .empty()) pdatbuf.U1_d  = std::make_unique<sycl::buffer<f64>  >(pdat.U1_d .data(),pdat.U1_d .size());
    if(! pdat.U3_s .empty()) pdatbuf.U3_s  = std::make_unique<sycl::buffer<f32_3>>(pdat.U3_s .data(),pdat.U3_s .size());
    if(! pdat.U3_d .empty()) pdatbuf.U3_d  = std::make_unique<sycl::buffer<f64_3>>(pdat.U3_d .data(),pdat.U3_d .size());

    return pdatbuf;
}
