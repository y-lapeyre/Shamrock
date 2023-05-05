// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/random/random.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/tree/RadixTree::serialize", testradixtreeserialize, 1) {



    u32 cnt = 1000;
    shammath::CoordRange<f32_3> range_coord{f32_3{0, 0, 0}, f32_3{1, 1, 1}};

    sycl::queue &q = shamsys::instance::get_compute_queue();

    auto buf = shamalgs::random::mock_buffer(0x1111, cnt, range_coord.lower, range_coord.upper);

    using u_morton = u32;

    RadixTree<u_morton, f32_3, 3> tree (
        q, 
        {range_coord.lower, range_coord.upper},
        buf, 
        cnt, 0);

    shamalgs::SerializeHelper ser;
    ser.allocate(tree.serialize_byte_size());
    tree.serialize(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(std::move(recov));

        RadixTree<u_morton, f32_3, 3> outser = RadixTree<u_morton, f32_3, 3>::deserialize(ser2);

        shamtest::asserts().assert_bool("input match out", outser == tree);
    }

}