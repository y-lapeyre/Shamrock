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
#include "shamrock/legacy/algs/sycl/basic/basic.hpp"
#include "shamrock/legacy/algs/sycl/defs.hpp"

namespace shamrock::tree {

    class TreeStructure {

        public:
        u32 internal_cell_count;

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
        build(sycl::queue &queue, u32 _internal_cell_count, sycl::buffer<T> &morton_buf) {
            internal_cell_count = _internal_cell_count;

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

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;
            sum += sizeof(internal_cell_count);

            auto add_ptr = [&](auto & a){
                if(a){
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_lchild_id           );
            add_ptr(buf_rchild_id           );
            add_ptr(buf_lchild_flag         );
            add_ptr(buf_rchild_flag         );
            add_ptr(buf_endrange            );

            return sum;
        }

        inline friend bool operator==(const TreeStructure &t1, const TreeStructure &t2) {
            bool cmp = true;

            cmp = cmp && (t1.internal_cell_count == t2.internal_cell_count);

            cmp = cmp && syclalgs::reduction::equals(*t1.buf_lchild_id   , *t2.buf_lchild_id   , t1.internal_cell_count);
            cmp = cmp && syclalgs::reduction::equals(*t1.buf_rchild_id   , *t2.buf_rchild_id   , t1.internal_cell_count);
            cmp = cmp && syclalgs::reduction::equals(*t1.buf_lchild_flag , *t2.buf_lchild_flag , t1.internal_cell_count);
            cmp = cmp && syclalgs::reduction::equals(*t1.buf_rchild_flag , *t2.buf_rchild_flag , t1.internal_cell_count);
            cmp = cmp && syclalgs::reduction::equals(*t1.buf_endrange    , *t2.buf_endrange    , t1.internal_cell_count);

            return cmp;
        }

        inline TreeStructure() = default;

        inline TreeStructure(const TreeStructure &other)
            : internal_cell_count(other.internal_cell_count),
              buf_lchild_id(syclalgs::basic::duplicate(other.buf_lchild_id)),     // size = internal
              buf_rchild_id(syclalgs::basic::duplicate(other.buf_rchild_id)),     // size = internal
              buf_lchild_flag(syclalgs::basic::duplicate(other.buf_lchild_flag)), // size = internal
              buf_rchild_flag(syclalgs::basic::duplicate(other.buf_rchild_flag)), // size = internal
              buf_endrange(syclalgs::basic::duplicate(other.buf_endrange))        // size = internal
        {}


    };

} // namespace shamrock::tree