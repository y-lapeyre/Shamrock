// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/memory/memory.hpp"
#include "shambase/sycl.hpp"

namespace shamrock::tree {

    template<class ipos_t, class pos_t>
    class TreeCellRanges {

        public:

        //this one is not used, it should be removed
        std::unique_ptr<sycl::buffer<ipos_t>>    buf_pos_min_cell;     // size = total count //rename to ipos
        std::unique_ptr<sycl::buffer<ipos_t>>    buf_pos_max_cell;     // size = total count

        //optional
        std::unique_ptr<sycl::buffer<pos_t>>     buf_pos_min_cell_flt; // size = total count //drop the flt part
        std::unique_ptr<sycl::buffer<pos_t>>     buf_pos_max_cell_flt; // size = total count

        inline bool are_range_int_built(){
            return bool(buf_pos_min_cell) && bool(buf_pos_max_cell);
        }

        inline bool are_range_float_built(){
            return bool(buf_pos_min_cell_flt) && bool(buf_pos_max_cell_flt);
        }

        inline TreeCellRanges() = default;

        inline TreeCellRanges(const TreeCellRanges &other)
            : 
              buf_pos_min_cell        (shamalgs::memory::duplicate(other.buf_pos_min_cell       )),     // size = total count
                buf_pos_max_cell        (shamalgs::memory::duplicate(other.buf_pos_max_cell       )),     // size = total count
                buf_pos_min_cell_flt    (shamalgs::memory::duplicate(other.buf_pos_min_cell_flt   )), // size = total count
                buf_pos_max_cell_flt    (shamalgs::memory::duplicate(other.buf_pos_max_cell_flt   )) // size = total count
        {}

        [[nodiscard]] inline u64 memsize() const {
            u64 sum = 0;

            auto add_ptr = [&](auto &a) {
                if (a) {
                    sum += a->byte_size();
                }
            };

            add_ptr(buf_pos_min_cell);
            add_ptr(buf_pos_max_cell);
            add_ptr(buf_pos_min_cell_flt);
            add_ptr(buf_pos_max_cell_flt);

            return sum;
        }
    };

}