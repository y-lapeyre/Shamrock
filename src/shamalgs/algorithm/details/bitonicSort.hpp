// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"


namespace shamalgs::algorithm::details {

    template<class Tkey, class Tval>
    void sort_by_key_bitonic_legacy(sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    template<class Tkey, class Tval, u32 MaxStencilSize>
    void sort_by_key_bitonic_updated(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len
    );

}