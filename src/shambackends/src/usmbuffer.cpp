// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file usmbuffer.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/usmbuffer.hpp"

namespace sham {

    template<USMKindTarget target>
    usmptr_holder<target>::usmptr_holder(size_t sz, sycl::queue &q) : queue(q), size(sz) {
        if constexpr (target == device) {
            usm_ptr = sycl::malloc_device(sz, queue);
        } else if constexpr (target == shared) {
            usm_ptr = sycl::malloc_shared(sz, queue);
        } else if constexpr (target == host) {
            usm_ptr = sycl::malloc_host(sz, queue);
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<USMKindTarget target>
    usmptr_holder<target>::~usmptr_holder() {
        sycl::free(usm_ptr, queue);
    }

    template class usmptr_holder<device>;
    template class usmptr_holder<shared>;
    template class usmptr_holder<host>;

} // namespace sham