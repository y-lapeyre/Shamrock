// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamalgs/memory/memory.hpp"
#include "shamalgs/reduction/reduction.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/tree/RadixTreeMortonBuilder.hpp"

namespace shamrock::tree {

    template<class u_morton>
    class TreeMortonCodes {
        public:
        
        u32 obj_cnt;
        
        std::unique_ptr<sycl::buffer<u_morton>> buf_morton;
        std::unique_ptr<sycl::buffer<u32>> buf_particle_index_map;

        template<class T>
        inline void build(
            sycl::queue &queue,
            shammath::CoordRange<T> coord_range,
            u32 obj_cnt,
            sycl::buffer<T> &pos_buf
        ) {

            this->obj_cnt = obj_cnt;

            using TProp = shambase::sycl_utils::VectorProperties<T>;

            RadixTreeMortonBuilder<u_morton, T, TProp::dimension>::build(
                queue,
                {coord_range.lower, coord_range.upper},
                pos_buf,
                obj_cnt,
                buf_morton,
                buf_particle_index_map
            );
        }

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

        sum += sizeof(obj_cnt);

            add_ptr(buf_morton);
            add_ptr(buf_particle_index_map);

            return sum;
        }

        inline TreeMortonCodes() = default;

        inline TreeMortonCodes(const TreeMortonCodes &other)
            : obj_cnt(other.obj_cnt),
              buf_morton(shamalgs::memory::duplicate(other.buf_morton)),
              buf_particle_index_map(shamalgs::memory::duplicate(other.buf_particle_index_map)) 
        {}


        inline friend bool operator==(const TreeMortonCodes &t1, const TreeMortonCodes &t2) {
            bool cmp = true;

            cmp = cmp && (t1.obj_cnt == t2.obj_cnt);

            cmp = cmp && shamalgs::reduction::equals(
                             *t1.buf_morton, *t2.buf_morton, t1.buf_morton->size()
                         );
            cmp = cmp && shamalgs::reduction::equals(
                             *t1.buf_particle_index_map, *t2.buf_particle_index_map, t1.buf_particle_index_map->size()
                         );

            return cmp;
        }
    };

} // namespace shamrock::tree