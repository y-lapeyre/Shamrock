// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/patchdata_layout.hpp"
#include "sph/kernels.hpp"
#include "sph/sphpart.hpp"
#include "tree/radix_tree.hpp"

namespace impl {

    template<class flt>
    inline void sycl_init_h_iter_bufs(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h

        );

    template<>
    inline void sycl_init_h_iter_bufs<f32>(
        sycl::queue & queue, 
        u32 or_element_cnt,
        
        u32 ihpart,
        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<f32> & hnew,
        sycl::buffer<f32> & omega,
        sycl::buffer<f32> & eps_h

        ){

        sycl::range range_npart{or_element_cnt};

        queue.submit([&](sycl::handler &cgh) {
            
            auto acc_hpart = pdat_buf_merge.fields_f32[ihpart]->get_access<sycl::access::mode::read>(cgh);
            auto eps = eps_h.get_access<sycl::access::mode::discard_write>(cgh);
            auto h    = hnew.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Init_iterate_h>( range_npart, [=](sycl::item<1> item) {
                    
                u32 id_a = (u32) item.get_id(0);

                h[id_a] = acc_hpart[id_a];
                eps[id_a] = 100;

            });

        });
    }



    template<class morton_prec, class Kernel>
    class IntSmoothingLenghtCompute{

        template<class flt>
        static void sycl_h_iter_step(
            sycl::queue & queue,
            u32 or_element_cnt,
            
            u32 ihpart,
            u32 ixyz,

            flt gpart_mass,
            flt htol_up_tol,
            flt htol_up_iter,

            Radix_Tree<morton_prec, sycl::vec<flt,3>> & radix_t,

            PatchDataBuffer & pdat_buf_merge,
            sycl::buffer<flt> & hnew,
            sycl::buffer<flt> & omega,
            sycl::buffer<flt> & eps_h
        );

    };


    

}