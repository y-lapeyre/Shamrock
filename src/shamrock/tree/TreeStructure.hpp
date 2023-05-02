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
#include "shamalgs/memory/memory.hpp"
#include "shamrock/legacy/algs/sycl/basic/basic.hpp"
#include "shamrock/legacy/algs/sycl/defs.hpp"
#include "shambase/exception.hpp"

namespace shamrock::tree {

    template<class u_morton>
    class TreeStructure {

        public:
        u32 internal_cell_count;
        bool one_cell_mode = false;

        std::unique_ptr<sycl::buffer<u32>> buf_lchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_rchild_id;  // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_lchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u8>> buf_rchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u32>> buf_endrange;   // size = internal (+1 if one cell mode)

        bool is_built() {
            return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag) &&
                   bool(buf_rchild_flag) && bool(buf_endrange);
        }

        inline void
        build(sycl::queue &queue, u32 _internal_cell_count, sycl::buffer<u_morton> &morton_buf) {

            if (!(_internal_cell_count < morton_buf.size())) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "morton buf must be at least with size() greater than internal_cell_count"
                );
            }

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

            one_cell_mode = false;
        }

        inline void build_one_cell_mode() {
            internal_cell_count = 1;
            buf_lchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_rchild_id       = std::make_unique<sycl::buffer<u32>>(internal_cell_count);
            buf_lchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_rchild_flag     = std::make_unique<sycl::buffer<u8>>(internal_cell_count);
            buf_endrange        = std::make_unique<sycl::buffer<u32>>(internal_cell_count+1); //this +1 is the signature of the one cell mode

            {
                sycl::host_accessor rchild_id{*buf_rchild_id, sycl::write_only, sycl::no_init};
                sycl::host_accessor lchild_id{*buf_lchild_id, sycl::write_only, sycl::no_init};
                sycl::host_accessor rchild_flag{*buf_rchild_flag, sycl::write_only, sycl::no_init};
                sycl::host_accessor lchild_flag{*buf_lchild_flag, sycl::write_only, sycl::no_init};
                sycl::host_accessor endrange{*buf_endrange, sycl::write_only, sycl::no_init};

                lchild_id[0]   = 0;
                rchild_id[0]   = 1;
                lchild_flag[0] = 1;
                rchild_flag[0] = 1;

                endrange[0]    = 1;
            }
            one_cell_mode = true;
        }

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;
            sum += sizeof(internal_cell_count);
            sum += sizeof(one_cell_mode);

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_lchild_id);
            add_ptr(buf_rchild_id);
            add_ptr(buf_lchild_flag);
            add_ptr(buf_rchild_flag);
            add_ptr(buf_endrange);

            return sum;
        }

        inline friend bool operator==(const TreeStructure &t1, const TreeStructure &t2) {
            bool cmp = true;

            cmp = cmp && (t1.internal_cell_count == t2.internal_cell_count);

            cmp = cmp && syclalgs::reduction::equals(
                             *t1.buf_lchild_id, *t2.buf_lchild_id, t1.internal_cell_count
                         );
            cmp = cmp && syclalgs::reduction::equals(
                             *t1.buf_rchild_id, *t2.buf_rchild_id, t1.internal_cell_count
                         );
            cmp = cmp && syclalgs::reduction::equals(
                             *t1.buf_lchild_flag, *t2.buf_lchild_flag, t1.internal_cell_count
                         );
            cmp = cmp && syclalgs::reduction::equals(
                             *t1.buf_rchild_flag, *t2.buf_rchild_flag, t1.internal_cell_count
                         );
            cmp = cmp && syclalgs::reduction::equals(
                             *t1.buf_endrange, *t2.buf_endrange, t1.internal_cell_count
                         );
            cmp = cmp && (t1.one_cell_mode == t2.one_cell_mode);

            return cmp;
        }

        inline TreeStructure() = default;

        inline TreeStructure(const TreeStructure &other)
            : internal_cell_count(other.internal_cell_count), one_cell_mode(other.one_cell_mode),
              buf_lchild_id(shamalgs::memory::duplicate(other.buf_lchild_id)),     // size = internal
              buf_rchild_id(shamalgs::memory::duplicate(other.buf_rchild_id)),     // size = internal
              buf_lchild_flag(shamalgs::memory::duplicate(other.buf_lchild_flag)), // size = internal
              buf_rchild_flag(shamalgs::memory::duplicate(other.buf_rchild_flag)), // size = internal
              buf_endrange(shamalgs::memory::duplicate(other.buf_endrange))        // size = internal
        {}

        inline TreeStructure(
            u32 internal_cell_count,
            bool one_cell_mode,
            std::unique_ptr<sycl::buffer<u32>> && buf_lchild_id,  
            std::unique_ptr<sycl::buffer<u32>> && buf_rchild_id,  
            std::unique_ptr<sycl::buffer<u8>> && buf_lchild_flag, 
            std::unique_ptr<sycl::buffer<u8>> && buf_rchild_flag, 
            std::unique_ptr<sycl::buffer<u32>> && buf_endrange
        ):
        internal_cell_count(internal_cell_count),
        one_cell_mode(one_cell_mode),
        buf_lchild_id(std::move(buf_lchild_id)),
        buf_rchild_id(std::move(buf_rchild_id)),
        buf_lchild_flag(std::move(buf_lchild_flag)),
        buf_rchild_flag(std::move(buf_rchild_flag)),
        buf_endrange(std::move(buf_endrange))
        {}
    };

} // namespace shamrock::tree