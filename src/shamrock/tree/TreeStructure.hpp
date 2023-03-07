// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "kernels/karras_alg.hpp"

namespace shamrock::tree {

    class TreeStructure {

        public:
        std::unique_ptr<sycl::buffer<u32>> buf_lchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_rchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_lchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_rchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_endrange;   // size = internal

        bool is_built() {
            return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag) &&
                   bool(buf_rchild_flag) && bool(buf_endrange);
        }

        template<class T>
        inline void
        build(sycl::queue &queue, u32 internal_cell_count, sycl::buffer<T> &morton_buf) {
            buf_lchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_rchild_id   = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_lchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_rchild_flag = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_endrange    = std::make_unique<sycl::buffer<u32>>(internal_cell_count);

            sycl_karras_alg(
                queue,
                internal_cell_count,
                morton_buf,
                *buf_lchild_id,
                *buf_rchild_id,
                *buf_lchild_flag,
                *buf_rchild_flag,
                *buf_endrange
            );
        }
    };

} // namespace shamrock::tree