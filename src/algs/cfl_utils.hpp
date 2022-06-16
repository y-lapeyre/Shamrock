// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "core/utils/syclreduction.hpp"
#include "aliases.hpp"
#include "core/patch/patchdata_buffer.hpp"
template<class flt>
class CflUtility{public:

    template<class LambdaCFL>
    inline static flt basic_cfl(PatchDataBuffer &pdat_buf,LambdaCFL && lambda_internal){

        SyCLHandler &hndl = SyCLHandler::get_instance();

        u32 npart_patch = pdat_buf.element_count;

        std::unique_ptr<sycl::buffer<flt>> buf_cfl = std::make_unique<sycl::buffer<flt>>(npart_patch);

        sycl::range<1> range_npart{npart_patch};

        auto ker_reduc_step_mincfl = [&](sycl::handler &cgh) {
            auto arr = buf_cfl->template get_access<sycl::access::mode::discard_write>(cgh);

            lambda_internal(cgh,*buf_cfl, range_npart);
        };

        hndl.get_queue_compute(0).submit(ker_reduc_step_mincfl);

        flt min_cfl = syclalg::get_min<flt>(hndl.get_queue_compute(0), buf_cfl);

        return min_cfl;

    }

};
