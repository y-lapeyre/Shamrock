// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include "shambase/sycl.hpp"
#include "shamrock/tree/TreeTraversal.hpp"
#include <array>

namespace shammodels::zeus  {

    template<class Tvec>
    struct OrientedNeighFaceList {
        shamrock::tree::ObjectCache neigh_info;
        Tvec normal;
    };

    template<class Tvec>
    struct NeighFaceList{
        std::array<OrientedNeighFaceList<Tvec>, 6> faces_lists;

        static constexpr u32 i_xm = 0 ;
        static constexpr u32 i_xp = 1 ;
        static constexpr u32 i_ym = 2 ;
        static constexpr u32 i_yp = 3 ;
        static constexpr u32 i_zm = 4 ;
        static constexpr u32 i_zp = 5 ;
        
    };

} // namespace shammodels::zeus
