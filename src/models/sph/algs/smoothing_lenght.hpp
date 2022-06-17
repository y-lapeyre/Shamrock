// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/patch/patchdata_buffer.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "models/sph/base/kernels.hpp"
#include "models/sph/algs/smoothing_lenght_impl.hpp"
#include "models/sph/base/sphpart.hpp"
#include "core/tree/radix_tree.hpp"

namespace sph {
namespace algs {

template<class flt, class morton_prec, class Kernel>
class SmoothingLenghtCompute{

    using vec = sycl::vec<flt, 3>;

    using Rtree = Radix_Tree<morton_prec, vec>;
    using Rta = walker::Radix_tree_accessor<morton_prec, vec>;

    flt htol_up_tol;
    flt htol_up_iter;

    u32 ihpart;
    u32 ixyz;




    public:

    SmoothingLenghtCompute (
        PatchDataLayout &pdl,
        f32 htol_up_tol,
        f32 htol_up_iter){

        ixyz = pdl.get_field_idx<vec>("xyz");
        ihpart = pdl.get_field_idx<flt>("hpart");

        this->htol_up_tol  = htol_up_tol;
        this->htol_up_iter = htol_up_iter;
    }

    inline void iterate_smoothing_lenght(
        sycl::queue & queue,
        u32 or_element_cnt,
        
        flt gpart_mass,

        Rtree & radix_t,

        PatchDataBuffer & pdat_buf_merge,
        sycl::buffer<flt> & hnew,
        sycl::buffer<flt> & omega,
        sycl::buffer<flt> & eps_h){

        impl::sycl_init_h_iter_bufs(queue, or_element_cnt,ihpart, pdat_buf_merge, hnew, omega, eps_h);

        for (u32 it_num = 0 ; it_num < 30; it_num++) {

            impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template sycl_h_iter_step<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);
        }

        impl::IntSmoothingLenghtCompute<morton_prec, Kernel>::template sycl_h_iter_omega<flt>(queue, 
                or_element_cnt, 
                ihpart, 
                ixyz,
                gpart_mass,
                htol_up_tol,
                htol_up_iter,
                radix_t,
                pdat_buf_merge, 
                hnew, 
                omega, 
                eps_h);

    }

    

};

} // namespace algs
} // namespace sph



