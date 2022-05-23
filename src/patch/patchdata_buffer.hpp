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
#include "patch/patchdata_field.hpp"
#include "patch/patchdata_layout.hpp"
#include "sys/sycl_handler.hpp"

/**
 * @brief sycl buffer loaded version of PatchData
 * 
 */
class PatchDataBuffer{ public:

    u32 element_count;
    
    PatchDataLayout & pdl;

    std::vector<std::unique_ptr<sycl::buffer<f32   >>> fields_f32;
    std::vector<std::unique_ptr<sycl::buffer<f32_2 >>> fields_f32_2;
    std::vector<std::unique_ptr<sycl::buffer<f32_3 >>> fields_f32_3;
    std::vector<std::unique_ptr<sycl::buffer<f32_4 >>> fields_f32_4;
    std::vector<std::unique_ptr<sycl::buffer<f32_8 >>> fields_f32_8;
    std::vector<std::unique_ptr<sycl::buffer<f32_16>>> fields_f32_16;

    std::vector<std::unique_ptr<sycl::buffer<f64   >>> fields_f64;
    std::vector<std::unique_ptr<sycl::buffer<f64_2 >>> fields_f64_2;
    std::vector<std::unique_ptr<sycl::buffer<f64_3 >>> fields_f64_3;
    std::vector<std::unique_ptr<sycl::buffer<f64_4 >>> fields_f64_4;
    std::vector<std::unique_ptr<sycl::buffer<f64_8 >>> fields_f64_8;
    std::vector<std::unique_ptr<sycl::buffer<f64_16>>> fields_f64_16;

    std::vector<std::unique_ptr<sycl::buffer<u32   >>> fields_u32;

    std::vector<std::unique_ptr<sycl::buffer<u64   >>> fields_u64;

    inline PatchDataBuffer(PatchDataLayout & pdl) : pdl(pdl) {}

    inline PatchDataBuffer(PatchDataLayout & pdl, u32 cnt) : pdl(pdl) {

        element_count = u32(cnt);


        for(u32 idx = 0; idx < pdl.fields_f32.size(); idx++){
            std::unique_ptr<sycl::buffer<f32>> buf;

                buf = std::make_unique<sycl::buffer<f32>>(pdl.fields_f32[idx].nvar * element_count);
            

            fields_f32.push_back(std::move(buf));
        }

        
        for(u32 idx = 0; idx <  pdl.fields_f32_2.size(); idx++){
            std::unique_ptr<sycl::buffer<f32_2>> buf;

                buf = std::make_unique<sycl::buffer<f32_2>>(pdl.fields_f32_2[idx].nvar * element_count);
            

            fields_f32_2.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f32_3.size(); idx++){
            std::unique_ptr<sycl::buffer<f32_3>> buf;

                buf = std::make_unique<sycl::buffer<f32_3>>(pdl.fields_f32_3[idx].nvar * element_count);
            

            fields_f32_3.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f32_4.size(); idx++){
            std::unique_ptr<sycl::buffer<f32_4>> buf;

                buf = std::make_unique<sycl::buffer<f32_4>>(pdl.fields_f32_4[idx].nvar * element_count);
            

            fields_f32_4.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f32_8.size(); idx++){
            std::unique_ptr<sycl::buffer<f32_8>> buf;

                buf = std::make_unique<sycl::buffer<f32_8>>(pdl.fields_f32_8[idx].nvar * element_count);
            

            fields_f32_8.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f32_16.size(); idx++){
            std::unique_ptr<sycl::buffer<f32_16>> buf;

                buf = std::make_unique<sycl::buffer<f32_16>>(pdl.fields_f32_16[idx].nvar * element_count);
            

            fields_f32_16.push_back(std::move(buf));
        }
        




        for(u32 idx = 0; idx <  pdl.fields_f64.size(); idx++){
            std::unique_ptr<sycl::buffer<f64>> buf;

                buf = std::make_unique<sycl::buffer<f64>>(pdl.fields_f64[idx].nvar * element_count);
            

            fields_f64.push_back(std::move(buf));
        }

        
        for(u32 idx = 0; idx <  pdl.fields_f64_2.size(); idx++){
            std::unique_ptr<sycl::buffer<f64_2>> buf;

                buf = std::make_unique<sycl::buffer<f64_2>>(pdl.fields_f64_2[idx].nvar * element_count);
            

            fields_f64_2.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f64_3.size(); idx++){
            std::unique_ptr<sycl::buffer<f64_3>> buf;

                buf = std::make_unique<sycl::buffer<f64_3>>(pdl.fields_f64_3[idx].nvar * element_count);
            

            fields_f64_3.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f64_4.size(); idx++){
            std::unique_ptr<sycl::buffer<f64_4>> buf;

                buf = std::make_unique<sycl::buffer<f64_4>>(pdl.fields_f64_4[idx].nvar * element_count);
            

            fields_f64_4.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f64_8.size(); idx++){
            std::unique_ptr<sycl::buffer<f64_8>> buf;

                buf = std::make_unique<sycl::buffer<f64_8>>(pdl.fields_f64_8[idx].nvar * element_count);
            

            fields_f64_8.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_f64_16.size(); idx++){
            std::unique_ptr<sycl::buffer<f64_16>> buf;

                buf = std::make_unique<sycl::buffer<f64_16>>(pdl.fields_f64_16[idx].nvar * element_count);
            

            fields_f64_16.push_back(std::move(buf));
        }





        for(u32 idx = 0; idx <  pdl.fields_u32.size(); idx++){
            std::unique_ptr<sycl::buffer<u32>> buf;

                buf = std::make_unique<sycl::buffer<u32>>(pdl.fields_u32[idx].nvar * element_count);
            

            fields_u32.push_back(std::move(buf));
        }

        for(u32 idx = 0; idx <  pdl.fields_u64.size(); idx++){
            std::unique_ptr<sycl::buffer<u64>> buf;

            
            buf = std::make_unique<sycl::buffer<u64>>(pdl.fields_u64[idx].nvar * element_count);
            

            fields_u64.push_back(std::move(buf));
        }


    }

};



inline PatchDataBuffer attach_to_patchData(PatchData & pdat){
    PatchDataBuffer pdatbuf(pdat.patchdata_layout);
    
    pdatbuf.element_count = u32(pdat.get_obj_cnt());

    std::cout << "attach to pdat : " << pdatbuf.element_count << std::endl;


    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32.size(); idx++){
        std::unique_ptr<sycl::buffer<f32>> buf;

        if(pdat.fields_f32[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32>>(pdat.fields_f32[idx].data(),pdat.fields_f32[idx].size());
        }

        pdatbuf.fields_f32.push_back({std::move(buf)});
    }

    
    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_2.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_2>> buf;

        if(pdat.fields_f32_2[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_2>>(pdat.fields_f32_2[idx].data(),pdat.fields_f32_2[idx].size());
        }

        pdatbuf.fields_f32_2.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_3.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_3>> buf;

        if(pdat.fields_f32_3[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_3>>(pdat.fields_f32_3[idx].data(),pdat.fields_f32_3[idx].size());
        }

        pdatbuf.fields_f32_3.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_4.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_4>> buf;

        if(pdat.fields_f32_4[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_4>>(pdat.fields_f32_4[idx].data(),pdat.fields_f32_4[idx].size());
        }

        pdatbuf.fields_f32_4.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_8.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_8>> buf;

        if(pdat.fields_f32_8[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_8>>(pdat.fields_f32_8[idx].data(),pdat.fields_f32_8[idx].size());
        }

        pdatbuf.fields_f32_8.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f32_16.size(); idx++){
        std::unique_ptr<sycl::buffer<f32_16>> buf;

        if(pdat.fields_f32_16[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f32_16>>(pdat.fields_f32_16[idx].data(),pdat.fields_f32_16[idx].size());
        }

        pdatbuf.fields_f32_16.push_back({std::move(buf)});
    }
    




    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64.size(); idx++){
        std::unique_ptr<sycl::buffer<f64>> buf;

        if(pdat.fields_f64[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64>>(pdat.fields_f64[idx].data(),pdat.fields_f64[idx].size());
        }

        pdatbuf.fields_f64.push_back({std::move(buf)});
    }

    
    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_2.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_2>> buf;

        if(pdat.fields_f64_2[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_2>>(pdat.fields_f64_2[idx].data(),pdat.fields_f64_2[idx].size());
        }

        pdatbuf.fields_f64_2.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_3.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_3>> buf;

        if(pdat.fields_f64_3[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_3>>(pdat.fields_f64_3[idx].data(),pdat.fields_f64_3[idx].size());
        }

        pdatbuf.fields_f64_3.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_4.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_4>> buf;

        if(pdat.fields_f64_4[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_4>>(pdat.fields_f64_4[idx].data(),pdat.fields_f64_4[idx].size());
        }

        pdatbuf.fields_f64_4.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_8.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_8>> buf;

        if(pdat.fields_f64_8[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_8>>(pdat.fields_f64_8[idx].data(),pdat.fields_f64_8[idx].size());
        }

        pdatbuf.fields_f64_8.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_f64_16.size(); idx++){
        std::unique_ptr<sycl::buffer<f64_16>> buf;

        if(pdat.fields_f64_16[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<f64_16>>(pdat.fields_f64_16[idx].data(),pdat.fields_f64_16[idx].size());
        }

        pdatbuf.fields_f64_16.push_back({std::move(buf)});
    }





    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_u32.size(); idx++){
        std::unique_ptr<sycl::buffer<u32>> buf;

        if(pdat.fields_u32[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<u32>>(pdat.fields_u32[idx].data(),pdat.fields_u32[idx].size());
        }

        pdatbuf.fields_u32.push_back({std::move(buf)});
    }

    for(u32 idx = 0; idx < pdat.patchdata_layout.fields_u64.size(); idx++){
        std::unique_ptr<sycl::buffer<u64>> buf;

        if(pdat.fields_u64[idx].get_obj_cnt() > 0){
            buf = std::make_unique<sycl::buffer<u64>>(pdat.fields_u64[idx].data(),pdat.fields_u64[idx].size());
        }

        pdatbuf.fields_u64.push_back({std::move(buf)});
    }







    return pdatbuf;
}
