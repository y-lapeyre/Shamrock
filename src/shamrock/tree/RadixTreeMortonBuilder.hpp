// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

template <class morton_t, class pos_t, u32 dim> class RadixTreeMortonBuilder {
    public:
    static void build(
        sycl::queue &queue,
        std::tuple<pos_t, pos_t> bounding_box,
        const std::unique_ptr<sycl::buffer<pos_t>> &pos_buf,
        u32 cnt_obj,
        std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton,
        std::unique_ptr<sycl::buffer<u32>> &out_buf_particle_index_map
    );
};