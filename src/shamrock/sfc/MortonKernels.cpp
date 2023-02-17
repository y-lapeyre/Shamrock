// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "MortonKernels.hpp"
#include "shamrock/math/integerManip.hpp"
#include "shamsys/legacy/log.hpp"

template <class T> class fill_trailling_buf;

template <class morton_t, class pos_t, u32 dim> class pos_to_morton;

template <class morton_t, class _pos_t, u32 dim> class irange_to_range;





namespace shamrock::sfc {

    template <class T>
    void details::sycl_fill_trailling_buffer(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<T>> &buf_morton
    ) {

        logger::debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        if (fill_count - morton_count == 0) {
            logger::debug_sycl_ln("MortonKernels", "sycl_fill_trailling_buffer skipping pow len 2 is ok");
            return;
        }

        sycl::range<1> range_npart{fill_count - morton_count};

        auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
            sycl::accessor m{*buf_morton, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<fill_trailling_buf<T>>(range_npart, [=](sycl::item<1> i) {
                
                m[morton_count + i.get_id()] = MortonInfo<T>::err_code;
                
            });
        };

        queue.submit(ker_fill_trailling_buf);
    }

    template void details::sycl_fill_trailling_buffer<u32>(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<u32>> &buf_morton
    );

    template void details::sycl_fill_trailling_buffer<u64>(
        sycl::queue &queue,
        u32 morton_count,
        u32 fill_count,
        std::unique_ptr<sycl::buffer<u64>> &buf_morton
    );

    template <class morton_t, class _pos_t, u32 dim>
    void MortonKernels<morton_t, _pos_t, dim>::sycl_xyz_to_morton(
        sycl::queue &queue,
        u32 pos_count,
        const std::unique_ptr<sycl::buffer<pos_t>> &in_positions,
        pos_t bounding_box_min,
        pos_t bounding_box_max,
        std::unique_ptr<sycl::buffer<morton_t>> &out_morton
    ) {

        logger::debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        sycl::range<1> range_cnt{pos_count};

        auto transf = get_transform(bounding_box_min, bounding_box_max);

        queue.submit([&](sycl::handler &cgh) {
            pos_t orig  = std::get<0>(transf);
            pos_t scale = std::get<1>(transf);

            sycl::accessor r{*in_positions, cgh, sycl::read_only};
            sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<pos_to_morton<morton_t, pos_t, dim>>(
                range_cnt,
                [=](sycl::item<1> item) {
                    int i = (int)item.get_id(0);

                    ipos_t mr = to_morton_grid(r[i], orig, scale);
                    m[i]      = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
                }
            );
        }

        );
    }

    template <class morton_t, class _pos_t, u32 dim>
    void MortonKernels<morton_t, _pos_t, dim>::sycl_irange_to_range(
        sycl::queue &queue,
        u32 buf_len,
        pos_t bounding_box_min,
        pos_t bounding_box_max,
        std::unique_ptr<sycl::buffer<ipos_t>> &buf_pos_min_cell,
        std::unique_ptr<sycl::buffer<ipos_t>> &buf_pos_max_cell,
        std::unique_ptr<sycl::buffer<pos_t>> &out_buf_pos_min_cell_flt,
        std::unique_ptr<sycl::buffer<pos_t>> &out_buf_pos_max_cell_flt
    ) {
        sycl::range<1> range_cell{buf_len};

        auto transf = get_transform(bounding_box_min, bounding_box_max);


        logger::debug_sycl_ln("MortonKernels", "submit : ", __PRETTY_FUNCTION__);

        auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
            pos_t orig  = std::get<0>(transf);
            pos_t scale = std::get<1>(transf);

            auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
            auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

            auto pos_min_cell_flt =
                sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
            auto pos_max_cell_flt =
                sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<irange_to_range<morton_t, pos_t, dim>>(
                range_cell,
                [=](sycl::item<1> item) {
                    u32 gid = (u32)item.get_id(0);

                    pos_min_cell_flt[gid] = to_real_space(pos_min_cell[gid], orig, scale);
                    pos_max_cell_flt[gid] = to_real_space(pos_max_cell[gid], orig, scale);
                }
            );
        };

        queue.submit(ker_convert_cell_ranges);
    }

    template class MortonKernels<u32, f32_3, 3>;
    template class MortonKernels<u64, f32_3, 3>;
    template class MortonKernels<u32, f64_3, 3>;
    template class MortonKernels<u64, f64_3, 3>;
    template class MortonKernels<u32, u32_3, 3>;
    template class MortonKernels<u64, u32_3, 3>;
    template class MortonKernels<u32, u64_3, 3>;
    template class MortonKernels<u64, u64_3, 3>;
} // namespace shamrock::sfc
