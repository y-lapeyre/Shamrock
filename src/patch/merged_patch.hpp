// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "interfaces/interface_handler.hpp"
#include "patchdata_buffer.hpp"
#include "patchscheduler/scheduler_mpi.hpp"

template<class vec>
struct MergedPatchDataBuffer {public:
    u32 or_element_cnt;
    std::unique_ptr<PatchDataBuffer> data;
    std::tuple<vec,vec> box;
};

template<class T>
struct MergedPatchCompFieldBuffer {public:
    u32 or_element_cnt;
    std::unique_ptr<sycl::buffer<T>> buf;
};




template<class pos_prec,class pos_vec>
inline void make_merge_patches(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,
    
    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){

    std::cout << "merging patches" << std::endl;

    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        auto tmp_box = sched.patch_data.sim_box.get_box<pos_prec>(cur_p);

        f32_3 min_box = std::get<0>(tmp_box);
        f32_3 max_box = std::get<1>(tmp_box);

        //std::cout << "patch : n°"<<id_patch << " -> making merge buf" << std::endl;

        u32 len_main = pdat_buf.element_count;

        u32 original_element = len_main;
        //merge_pdat_buf[id_patch].or_element_cnt = len_main;

        {
            const std::vector<std::tuple<u64, std::unique_ptr<PatchData>>> & p_interf_lst = interface_hndl.get_interface_list(id_patch);
            for (auto & [int_pid, pdat_ptr] : p_interf_lst) {


                u32 cnt = pdat_ptr->get_obj_cnt();
                //std::cout << "received interf : " << cnt << std::endl;
                len_main += (cnt);
            }
        }
        
        u32 total_element = len_main;
        //merge_pdat_buf[id_patch].data.element_count = len_main;

        

        
        std::unique_ptr<PatchDataBuffer> merged_buf = std::make_unique<PatchDataBuffer>(pdat_buf.pdl, total_element);



        std::vector<u32> fields_f32_offset;
        std::vector<u32> fields_f32_2_offset;
        std::vector<u32> fields_f32_3_offset;
        std::vector<u32> fields_f32_4_offset;
        std::vector<u32> fields_f32_8_offset;
        std::vector<u32> fields_f32_16_offset;
        std::vector<u32> fields_f64_offset;
        std::vector<u32> fields_f64_2_offset;
        std::vector<u32> fields_f64_3_offset;
        std::vector<u32> fields_f64_4_offset;
        std::vector<u32> fields_f64_8_offset;
        std::vector<u32> fields_f64_16_offset;
        std::vector<u32> fields_u32_offset;
        std::vector<u32> fields_u64_offset;







        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32.size(); idx++){
            fields_f32_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_2.size(); idx++){
            fields_f32_2_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_3.size(); idx++){
            fields_f32_3_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_4.size(); idx++){
            fields_f32_4_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_8.size(); idx++){
            fields_f32_8_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f32_16.size(); idx++){
            fields_f32_16_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64.size(); idx++){
            fields_f64_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_2.size(); idx++){
            fields_f64_2_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_3.size(); idx++){
            fields_f64_3_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_4.size(); idx++){
            fields_f64_4_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_8.size(); idx++){
            fields_f64_8_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_f64_16.size(); idx++){
            fields_f64_16_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_u32.size(); idx++){
            fields_u32_offset.push_back(0);
        }

        for(u32 idx = 0; idx <  pdat_buf.pdl.fields_u64.size(); idx++){
            fields_u64_offset.push_back(0);
        }





















        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_2.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32_2[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32_2[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_2_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_3.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32_3[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32_3[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_3_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_4.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32_4[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32_4[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_4_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_8.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32_8[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32_8[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_8_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_16.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f32_16[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f32_16[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f32_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f32_16_offset[idx] += pdat_buf.element_count  *  nvar ;
        }






        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_2.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64_2[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64_2[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_2_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_3.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64_3[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64_3[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_3_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_4.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64_4[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64_4[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_4_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_8.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64_8[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64_8[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_8_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_16.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_f64_16[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_f64_16[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_f64_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_f64_16_offset[idx] += pdat_buf.element_count  *  nvar ;
        }



        for(u32 idx = 0; idx < pdat_buf.pdl.fields_u32.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_u32[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_u32[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_u32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_u32_offset[idx] += pdat_buf.element_count  *  nvar ;
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_u64.size(); idx++){
            u32 nvar = merged_buf->pdl.fields_u64[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto source = pdat_buf.fields_u64[idx]->get_access<sycl::access::mode::read>(cgh);
                auto dest = merged_buf->fields_u64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
            fields_u64_offset[idx] += pdat_buf.element_count  *  nvar ;
        }








        

        

        interface_hndl.for_each_interface(
            id_patch, 
            hndl.get_queue_compute(0), 
            [&](u64 patch_id, u64 interf_patch_id, PatchDataBuffer & interfpdat, std::tuple<f32_3,f32_3> box){

                //std::cout <<  "patch : n°"<< id_patch << " -> interface : "<<interf_patch_id << " merging" << std::endl;

                min_box = sycl::min(std::get<0>(box),min_box);
                max_box = sycl::max(std::get<1>(box),max_box);







                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                    });
                    fields_f32_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32_2.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32_2[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32_2[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_2_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f32_2_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32_3.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32_3[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32_3[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_3_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f32_3_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32_4.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32_4[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32_4[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_4_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f32_4_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32_8.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32_8[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32_8[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_8_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f32_8_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f32_16.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f32_16[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f32_16[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f32_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f32_16_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f32_16_offset[idx] += interfpdat.element_count  *  nvar ;
                }






                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64_2.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64_2[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64_2[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_2_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_2_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64_3.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64_3[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64_3[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_3_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_3_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64_4.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64_4[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64_4[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_4_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_4_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64_8.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64_8[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64_8[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_8_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_8_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_f64_16.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_f64_16[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_f64_16[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_f64_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_f64_16_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_f64_16_offset[idx] += interfpdat.element_count  *  nvar ;
                }



                for(u32 idx = 0; idx < interfpdat.pdl.fields_u32.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_u32[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_u32[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_u32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_u32_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_u32_offset[idx] += interfpdat.element_count  *  nvar ;
                }

                for(u32 idx = 0; idx < interfpdat.pdl.fields_u64.size(); idx++){
                    u32 nvar = merged_buf->pdl.fields_u64[idx].nvar;
                    hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                        auto source = interfpdat.fields_u64[idx]->get_access<sycl::access::mode::read>(cgh);
                        auto dest = merged_buf->fields_u64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                        auto off = fields_u64_offset[idx];
                        cgh.parallel_for( sycl::range{interfpdat.element_count*nvar}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });                    });
                    fields_u64_offset[idx] += interfpdat.element_count  *  nvar ;
                }

            }
        );

        merge_pdat_buf[id_patch].or_element_cnt = original_element;
        merge_pdat_buf[id_patch].data = std::move(merged_buf);
        merge_pdat_buf[id_patch].box = {min_box,max_box};




    });


}



template<class pos_prec,class pos_vec>
inline void write_back_merge_patches(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,
    
    std::unordered_map<u64,MergedPatchDataBuffer<pos_vec>> & merge_pdat_buf){


    SyCLHandler &hndl = SyCLHandler::get_instance();

    sched.for_each_patch_buf([&](u64 id_patch, Patch cur_p, PatchDataBuffer & pdat_buf) {
        if(merge_pdat_buf.at(id_patch).or_element_cnt == 0) std::cout << " empty => skipping" << std::endl;


        std::cout << "patch : n°"<<id_patch << " -> write back merge buf" << std::endl;










        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_2.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32_2[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32_2[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_3.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32_3[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32_3[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_4.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32_4[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32_4[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_8.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32_8[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32_8[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f32_16.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f32_16[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f32_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f32_16[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }






        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_2.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64_2[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64_2[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64_2[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_3.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64_3[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64_3[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64_3[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_4.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64_4[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64_4[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64_4[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_8.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64_8[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64_8[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64_8[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

        for(u32 idx = 0; idx < pdat_buf.pdl.fields_f64_16.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_f64_16[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_f64_16[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_f64_16[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }



        for(u32 idx = 0; idx < pdat_buf.pdl.fields_u32.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_u32[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_u32[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_u32[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }


        for(u32 idx = 0; idx < pdat_buf.pdl.fields_u64.size(); idx++){
            u32 nvar = pdat_buf.pdl.fields_u64[idx].nvar;
            hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                auto dest = pdat_buf.fields_u64[idx]->get_access<sycl::access::mode::discard_write>(cgh);
                auto source = merge_pdat_buf.at(id_patch).data->fields_u64[idx]->template get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for( sycl::range{pdat_buf.element_count*nvar}, [=](sycl::item<1> item) { dest[item] = source[item]; });
            });
        }

    });

}



template<class pos_prec,class pos_vec,class T>
inline void make_merge_patches_comp_field(
    PatchScheduler & sched,
    InterfaceHandler<pos_vec, pos_prec> & interface_hndl,

    PatchComputeField<f32> & comp_field,
    PatchComputeFieldInterfaces<f32> & comp_field_interf,

    std::unordered_map<u64,MergedPatchCompFieldBuffer<T>> & merge_pdat_comp_field){



    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        auto & compfield_buf = comp_field.field_data_buf[id_patch];

        std::cout << "patch : n°"<<id_patch << " -> making merge comp field" << std::endl;

        u32 len_main = compfield_buf->size();
        merge_pdat_comp_field[id_patch].or_element_cnt = len_main;

        {
            
            const std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>> & p_interf_lst = comp_field_interf.interface_map[id_patch];
            for (auto & [int_pid, pdat_ptr] : p_interf_lst) {
                len_main += (pdat_ptr->size());
            }
        }


        merge_pdat_comp_field[id_patch].buf = std::make_unique<sycl::buffer<f32>>(len_main);


        u32 offset_buf = 0;

        
        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
            auto source = compfield_buf->get_access<sycl::access::mode::read>(cgh);
            auto dest = merge_pdat_comp_field[id_patch].buf->template get_access<sycl::access::mode::discard_write>(cgh);
            cgh.parallel_for( sycl::range{compfield_buf->size()}, [=](sycl::item<1> item) { dest[item] = source[item]; });
        });
        offset_buf += compfield_buf->size();
        

        std::vector<std::tuple<u64, std::unique_ptr<PatchDataField<T>>>> & p_interf_lst = comp_field_interf.interface_map[id_patch];

        for (auto & [int_pid, pdat_ptr] : p_interf_lst) {

            if(pdat_ptr->size() > 0){

                //std::cout <<  "patch : n°"<< id_patch << " -> interface : "<<interf_patch_id << " merging" << std::endl;
                sycl::buffer<T> tmp_buf = sycl::buffer<T>(pdat_ptr->data(),pdat_ptr->size());

                u32 len_int =  pdat_ptr->size();

                hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                    auto source = tmp_buf.template get_access<sycl::access::mode::read>(cgh);
                    auto dest = merge_pdat_comp_field[id_patch].buf->template get_access<sycl::access::mode::discard_write>(cgh);
                    auto off = offset_buf;
                    cgh.parallel_for( sycl::range{len_int}, [=](sycl::item<1> item) { dest[item.get_id(0) + off] = source[item]; });
                });
                offset_buf += len_int;
                
            }
        }



        
        

    });


}
