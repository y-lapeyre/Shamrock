// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "aliases.hpp"
#include "shambase/string.hpp"
#include "shamrock/legacy/patch/utility/serialpatchtree.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamsys/legacy/log.hpp"

template <class posvec, class kername>
inline sycl::buffer<u64> __compute_object_patch_owner(sycl::queue &queue, sycl::buffer<posvec> &position_buffer, u32 len,
                                                      SerialPatchTree<posvec> &sptree) {

    
    return sptree.compute_patch_owner(queue, position_buffer, len);
}