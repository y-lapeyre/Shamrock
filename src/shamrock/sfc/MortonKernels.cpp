#include "MortonKernels.hpp"
#include "shamrock/math/integerManip.hpp"
#include "shamsys/legacy/log.hpp"

using namespace shamrock::sfc;

static constexpr u32 err_code_32 = 4294967295U;
static constexpr u64 err_code_64 = 18446744073709551615UL;

template <>
void details::sycl_fill_trailling_buffer<u32, err_code_32>(
    sycl::queue &queue,
    u32 morton_count,
    u32 fill_count,
    std::unique_ptr<sycl::buffer<u32>> &buf_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_fill_trailling_buffer<u32>");

    if (fill_count - morton_count == 0) {
        std::cout << "skipping" << std::endl;
        return;
    }

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class fill_trailling_buf_u32>(range_npart, [=](sycl::item<1> i) {
            m[morton_count + i.get_id()] = err_code_32;
        });
    };

    queue.submit(ker_fill_trailling_buf);
}

template <>
void details::sycl_fill_trailling_buffer<u64, err_code_64>(
    sycl::queue &queue,
    u32 morton_count,
    u32 fill_count,
    std::unique_ptr<sycl::buffer<u64>> &buf_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_fill_trailling_buffer<u64>");

    if (fill_count - morton_count == 0) {
        std::cout << "skipping" << std::endl;
        return;
    }

    sycl::range<1> range_npart{fill_count - morton_count};

    auto ker_fill_trailling_buf = [&](sycl::handler &cgh) {
        auto m = buf_morton->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class fill_trailling_buf_u64>(range_npart, [=](sycl::item<1> i) {
            m[morton_count + i.get_id()] = err_code_64;
        });
    };

    queue.submit(ker_fill_trailling_buf);
}

///////////////////
//coordinate transforms
///////////////////

#if false

template <>
auto MortonKernels<u32, f32_3, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {

    constexpr coord_t grd_scale = Morton::val_count;

    return {bounding_box_min, (bounding_box_max - bounding_box_min)/grd_scale};
}

template <>
auto MortonKernels<u32, f64_3, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {

    constexpr coord_t grd_scale = Morton::val_count;

    return {bounding_box_min, (bounding_box_max - bounding_box_min)/grd_scale};
}

template <>
auto MortonKernels<u64, f32_3, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {

    constexpr coord_t grd_scale = Morton::val_count;

    return {bounding_box_min, (bounding_box_max - bounding_box_min)/grd_scale};
}

template <>
auto MortonKernels<u64, f64_3, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {
    constexpr coord_t grd_scale = Morton::val_count;

    return {bounding_box_min, (bounding_box_max - bounding_box_min)/grd_scale};
}






template <>
auto MortonKernels<u32, MortonCodes<u32, 3>::int_vec_repr, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {

    // to convert int coord to morton we need to have a cubic box
    // with 2^n side lenght >= max morton side lenght

    int_t dx = bounding_box_max.x() - bounding_box_min.x();
    int_t dy = bounding_box_max.y() - bounding_box_min.y();
    int_t dz = bounding_box_max.z() - bounding_box_min.z();

    if (!(dx == dy && dy == dz)) {
        throw std::invalid_argument("The bounding box is ot a cube");
    }

    // validity condition
    auto check_axis = [](int_t len) -> bool {
        bool is_pow2 = shamrock::math::int_manip::get_next_pow2_val(len) == len;

        if (len > Morton::max_val && is_pow2) {
            return true;
        }
        return false;
    };

    bool check_x = check_axis(dx);
    bool check_y = check_axis(dy);
    bool check_z = check_axis(dz);

    if (!check_x) {
        throw std::invalid_argument(
            "The x axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    if (!check_y) {
        throw std::invalid_argument(
            "The y axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    if (!check_z) {
        throw std::invalid_argument(
            "The z axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    // we can only use dx since they must be equal to reach this call
    int_t scale = dx / (Morton::val_count);

    return {bounding_box_min,pos_t{scale,scale,scale}};
}

template <>
auto MortonKernels<u64, MortonCodes<u64, 3>::int_vec_repr, 3>::get_transform(pos_t bounding_box_min, pos_t bounding_box_max)
    -> std::tuple<pos_t, pos_t> {
    // to convert int coord to morton we need to have a cubic box
    // with 2^n side lenght >= max morton side lenght

    int_t dx = bounding_box_max.x() - bounding_box_min.x();
    int_t dy = bounding_box_max.y() - bounding_box_min.y();
    int_t dz = bounding_box_max.z() - bounding_box_min.z();

    if (!(dx == dy && dy == dz)) {
        throw std::invalid_argument("The bounding box is ot a cube");
    }

    // validity condition
    auto check_axis = [](int_t len) -> bool {
        bool is_pow2 = shamrock::math::int_manip::get_next_pow2_val(len) == len;

        if (len > Morton::max_val && is_pow2) {
            return true;
        }
        return false;
    };

    bool check_x = check_axis(dx);
    bool check_y = check_axis(dy);
    bool check_z = check_axis(dz);

    if (!check_x) {
        throw std::invalid_argument(
            "The x axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    if (!check_y) {
        throw std::invalid_argument(
            "The y axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    if (!check_z) {
        throw std::invalid_argument(
            "The z axis bounding box is not a pow of 2 with lenght >= morton_max"
        );
    }

    // we can only use dx since they must be equal to reach this call
    int_t scale = dx / (Morton::val_count);

    return {bounding_box_min,pos_t{scale,scale,scale}};
}










template <>
auto MortonKernels<u32, f32_3, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

    pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::fmin(sycl::fmax(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::fmin(sycl::fmax(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::fmin(sycl::fmax(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
}

template <>
auto MortonKernels<u32, f64_3, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

    pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::fmin(sycl::fmax(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::fmin(sycl::fmax(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::fmin(sycl::fmax(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
}

template <>
auto MortonKernels<u64, f32_3, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

        pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::fmin(sycl::fmax(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::fmin(sycl::fmax(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::fmin(sycl::fmax(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};

}

template <>
auto MortonKernels<u64, f64_3, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

    pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::fmin(sycl::fmax(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::fmin(sycl::fmax(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::fmin(sycl::fmax(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
    
}


template <>
auto MortonKernels<u32, MortonCodes<u32, 3>::int_vec_repr, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

    pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::min(sycl::max(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::min(sycl::max(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::min(sycl::max(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
    
}

template <>
auto MortonKernels<u64, MortonCodes<u64, 3>::int_vec_repr, 3>::to_morton_grid(pos_t pos, pos_t origin, pos_t scale)
    -> ipos_t {

    pos_t unit_coord = (pos - origin)/scale;

    constexpr coord_t max_bound = Morton::max_val;
    constexpr coord_t zero = 0;

    unit_coord.x() = sycl::min(sycl::max(unit_coord.x() , zero), max_bound);
    unit_coord.y() = sycl::min(sycl::max(unit_coord.y() , zero), max_bound);
    unit_coord.z() = sycl::min(sycl::max(unit_coord.z() , zero), max_bound);

    return ipos_t{int_t(unit_coord.x()),int_t(unit_coord.y()),int_t(unit_coord.z())};
    
}

#endif





template <>
void MortonKernels<u32, f32_3, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> &in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> &out_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_xyz_to_morton<u32,f32_3,3>");

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class pos_to_morton_u32_f32_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    }

    );
}

template <>
void MortonKernels<u32, f64_3, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> &in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> &out_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_xyz_to_morton<u32,f64_3,3>");

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};


        cgh.parallel_for<class pos_to_morton_u32_f64_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    }

    );
}

template <>
void MortonKernels<u64, f32_3, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f32_3>> &in_positions,
    f32_3 bounding_box_min,
    f32_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> &out_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_xyz_to_morton<u64,f32_3,3>");

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};


        cgh.parallel_for<class pos_to_morton_u64_f32_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    }

    );
}

template <>
void MortonKernels<u64, f64_3, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<f64_3>> &in_positions,
    f64_3 bounding_box_min,
    f64_3 bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> &out_morton
) {

    logger::debug_sycl_ln("MortonKernels", "submit : sycl_xyz_to_morton<u64,f64_3,3>");

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};


        cgh.parallel_for<class pos_to_morton_u64_f64_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    }

    );
}

template <>
void MortonKernels<u32, MortonCodes<u32, 3>::int_vec_repr, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<MortonCodes<u32, 3>::int_vec_repr>> &in_positions,
    MortonCodes<u32, 3>::int_vec_repr bounding_box_min,
    MortonCodes<u32, 3>::int_vec_repr bounding_box_max,
    std::unique_ptr<sycl::buffer<u32>> &out_morton
) {

    logger::debug_sycl_ln(
        "MortonKernels", "submit : sycl_xyz_to_morton<u32,MortonCodes<u32, 3>::int_vec_repr,3>"
    );

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class pos_to_morton_u32_u16_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    });
}

template <>
void MortonKernels<u64, MortonCodes<u64, 3>::int_vec_repr, 3>::sycl_xyz_to_morton(
    sycl::queue &queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<MortonCodes<u64, 3>::int_vec_repr>> &in_positions,
    MortonCodes<u64, 3>::int_vec_repr bounding_box_min,
    MortonCodes<u64, 3>::int_vec_repr bounding_box_max,
    std::unique_ptr<sycl::buffer<u64>> &out_morton
) {

    logger::debug_sycl_ln(
        "MortonKernels", "submit : sycl_xyz_to_morton<u64,MortonCodes<u64, 3>::int_vec_repr,3>"
    );

    sycl::range<1> range_cnt{pos_count};

    auto transf = get_transform(bounding_box_min, bounding_box_max);

    queue.submit([&](sycl::handler &cgh) {
        pos_t orig = std::get<0>(transf);
        pos_t scale = std::get<1>(transf);

        sycl::accessor r{*in_positions, cgh, sycl::read_only};
        sycl::accessor m{*out_morton, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class pos_to_morton_u64_u32_3_3>(range_cnt, [=](sycl::item<1> item) {
            int i = (int)item.get_id(0);

            ipos_t mr = to_morton_grid(r[i], orig,scale);
            m[i] = Morton::icoord_to_morton(mr.x(), mr.y(), mr.z());
        });
    });
}








template <>
void MortonKernels<u32, f32_3, 3>::sycl_irange_to_range(
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

    

    constexpr f32 div = 1.f / f32(Morton::val_count);

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        pos_t b_box_min = bounding_box_min;
        pos_t b_box_max = bounding_box_max;

        auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
        auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

        auto pos_min_cell_flt =
            sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
        auto pos_max_cell_flt =
            sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class Convert_cell_range_u32_f32>(range_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0()) * div;
            pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0()) * div;

            pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1()) * div;
            pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1()) * div;

            pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2()) * div;
            pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2()) * div;

            pos_min_cell_flt[gid] *= b_box_max - b_box_min;
            pos_min_cell_flt[gid] += b_box_min;

            pos_max_cell_flt[gid] *= b_box_max - b_box_min;
            pos_max_cell_flt[gid] += b_box_min;
        });
    };

    queue.submit(ker_convert_cell_ranges);
}

template <>
void MortonKernels<u64, f32_3, 3>::sycl_irange_to_range(
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

    constexpr f32 div = 1.f / f32(Morton::val_count);

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        pos_t b_box_min = bounding_box_min;
        pos_t b_box_max = bounding_box_max;

        auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
        auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

        auto pos_min_cell_flt =
            sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
        auto pos_max_cell_flt =
            sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class Convert_cell_range_u64_f32>(range_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            pos_min_cell_flt[gid].s0() = f32(pos_min_cell[gid].s0()) * div;
            pos_max_cell_flt[gid].s0() = f32(pos_max_cell[gid].s0()) * div;

            pos_min_cell_flt[gid].s1() = f32(pos_min_cell[gid].s1()) * div;
            pos_max_cell_flt[gid].s1() = f32(pos_max_cell[gid].s1()) * div;

            pos_min_cell_flt[gid].s2() = f32(pos_min_cell[gid].s2()) * div;
            pos_max_cell_flt[gid].s2() = f32(pos_max_cell[gid].s2()) * div;

            pos_min_cell_flt[gid] *= b_box_max - b_box_min;
            pos_min_cell_flt[gid] += b_box_min;

            pos_max_cell_flt[gid] *= b_box_max - b_box_min;
            pos_max_cell_flt[gid] += b_box_min;
        });
    };

    queue.submit(ker_convert_cell_ranges);
}

template <>
void MortonKernels<u32, f64_3, 3>::sycl_irange_to_range(
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

    constexpr f64 div = 1.f / f64(Morton::val_count);

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        pos_t b_box_min = bounding_box_min;
        pos_t b_box_max = bounding_box_max;

        auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
        auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

        auto pos_min_cell_flt =
            sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
        auto pos_max_cell_flt =
            sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class Convert_cell_range_u32_f64>(range_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0()) * div;
            pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0()) * div;

            pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1()) * div;
            pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1()) * div;

            pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2()) * div;
            pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2()) * div;

            pos_min_cell_flt[gid] *= b_box_max - b_box_min;
            pos_min_cell_flt[gid] += b_box_min;

            pos_max_cell_flt[gid] *= b_box_max - b_box_min;
            pos_max_cell_flt[gid] += b_box_min;
        });
    };

    queue.submit(ker_convert_cell_ranges);
}

template <>
void MortonKernels<u64, f64_3, 3>::sycl_irange_to_range(
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

    constexpr f64 div = 1.f / f64(Morton::val_count);

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        pos_t b_box_min = bounding_box_min;
        pos_t b_box_max = bounding_box_max;

        auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
        auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

        auto pos_min_cell_flt =
            sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
        auto pos_max_cell_flt =
            sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class Convert_cell_range_u64_f64>(range_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0()) * div;
            pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0()) * div;

            pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1()) * div;
            pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1()) * div;

            pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2()) * div;
            pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2()) * div;

            pos_min_cell_flt[gid] *= b_box_max - b_box_min;
            pos_min_cell_flt[gid] += b_box_min;

            pos_max_cell_flt[gid] *= b_box_max - b_box_min;
            pos_max_cell_flt[gid] += b_box_min;
        });
    };

    queue.submit(ker_convert_cell_ranges);
}

template <>
void MortonKernels<u64, MortonCodes<u64, 3>::int_vec_repr, 3>::sycl_irange_to_range(
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

    constexpr f64 div = 1.f / f64(Morton::val_count);

    auto ker_convert_cell_ranges = [&](sycl::handler &cgh) {
        pos_t b_box_min = bounding_box_min;
        pos_t b_box_max = bounding_box_max;

        auto pos_min_cell = sycl::accessor{*buf_pos_min_cell, cgh, sycl::read_only};
        auto pos_max_cell = sycl::accessor{*buf_pos_max_cell, cgh, sycl::read_only};

        auto pos_min_cell_flt =
            sycl::accessor{*out_buf_pos_min_cell_flt, cgh, sycl::write_only, sycl::no_init};
        auto pos_max_cell_flt =
            sycl::accessor{*out_buf_pos_max_cell_flt, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for<class Convert_cell_range_u64_u32_3>(range_cell, [=](sycl::item<1> item) {
            u32 gid = (u32)item.get_id(0);

            pos_min_cell_flt[gid].s0() = f64(pos_min_cell[gid].s0()) * div;
            pos_max_cell_flt[gid].s0() = f64(pos_max_cell[gid].s0()) * div;

            pos_min_cell_flt[gid].s1() = f64(pos_min_cell[gid].s1()) * div;
            pos_max_cell_flt[gid].s1() = f64(pos_max_cell[gid].s1()) * div;

            pos_min_cell_flt[gid].s2() = f64(pos_min_cell[gid].s2()) * div;
            pos_max_cell_flt[gid].s2() = f64(pos_max_cell[gid].s2()) * div;

            pos_min_cell_flt[gid] *= b_box_max - b_box_min;
            pos_min_cell_flt[gid] += b_box_min;

            pos_max_cell_flt[gid] *= b_box_max - b_box_min;
            pos_max_cell_flt[gid] += b_box_min;
        });
    };

    queue.submit(ker_convert_cell_ranges);
}