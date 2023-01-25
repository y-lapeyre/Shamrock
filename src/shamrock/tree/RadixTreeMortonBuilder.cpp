// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée
// David--Cléris
// <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License,
// see LICENSE for more information
//
// -------------------------------------------------------//

#include "RadixTreeMortonBuilder.hpp"
#include "kernels/key_morton_sort.hpp"
#include "shamrock/math/integerManip.hpp"
#include "shamrock/sfc/morton.hpp"
#include "shamrock/tree/kernels/morton_kernels.hpp"
#include "shamsys/legacy/log.hpp"

using namespace logger;
using namespace shamrock::math::int_manip;

template <class morton_t, class pos_t, u32 dim>
void RadixTreeMortonBuilder<morton_t, pos_t, dim>::build(
    sycl::queue &queue,
    std::tuple<pos_t, pos_t> bounding_box,
    std::unique_ptr<sycl::buffer<pos_t>> &pos_buf,
    u32 cnt_obj,
    std::unique_ptr<sycl::buffer<morton_t>> &out_buf_morton,
    std::unique_ptr<sycl::buffer<u32>> &out_buf_particle_index_map
) {

    if (cnt_obj > i32_max - 1) {
        throw std::invalid_argument("number of element in patch above i32_max-1");
    }

    debug_sycl_ln("RadixTree", "box dim :", bounding_box);

    u32 morton_len = get_next_pow2_val(cnt_obj);

    debug_sycl_ln("RadixTree", "morton buffer lenght :", morton_len);
    out_buf_morton = std::make_unique<sycl::buffer<morton_t>>(morton_len);

    sycl_xyz_to_morton<morton_t, pos_t, dim>(
        queue,
        cnt_obj,
        pos_buf,
        std::get<0>(bounding_box),
        std::get<1>(bounding_box),
        out_buf_morton
    );

    sycl_fill_trailling_buffer<morton_t>(queue, cnt_obj, morton_len, out_buf_morton);

    out_buf_particle_index_map = std::make_unique<sycl::buffer<u32>>(morton_len);

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor pidm{*out_buf_particle_index_map, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range(morton_len), [=](sycl::item<1> item) {
            pidm[item] = item.get_id(0);
        });
    });

    sycl_sort_morton_key_pair(queue, morton_len, out_buf_particle_index_map, out_buf_morton);
}

template class RadixTreeMortonBuilder<u32, sycl::vec<f32, 3>, 3>;
template class RadixTreeMortonBuilder<u32, sycl::vec<f64, 3>, 3>;
template class RadixTreeMortonBuilder<u64, sycl::vec<f32, 3>, 3>;
template class RadixTreeMortonBuilder<u64, sycl::vec<f64, 3>, 3>;

using namespace shamrock::sfc;

template class RadixTreeMortonBuilder<u32, MortonCodes<u32, 3>::int_vec_repr, 3>;
template class RadixTreeMortonBuilder<u64, MortonCodes<u64, 3>::int_vec_repr, 3>;

/////
