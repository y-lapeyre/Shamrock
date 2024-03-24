// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shambackends/usmbuffer.hpp"


TestStart(Unittest, "shambackends/usmbuffer", usmbuffer_consttructor, 1){
    using namespace sham;

    constexpr USMKindTarget target = USMKindTarget::device;
    const size_t sz = 10;
    using T = int;

    sycl::queue & q = shamsys::instance::get_compute_queue();
    sham::usmbuffer<T,target> buf{sz,q};

    REQUIRE(buf.get_ptr() != nullptr);
    REQUIRE(buf.get_read_only_ptr() != nullptr);
    REQUIRE(buf.get_size() == sz);
    REQUIRE(buf.get_bytesize() == sz*sizeof(T));
    REQUIRE(buf.get_queue() == q);
}

