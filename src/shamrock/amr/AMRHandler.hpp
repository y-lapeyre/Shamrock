// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "AMRCell.hpp"
#include "aliases.hpp"
#include "shamrock/patch/PatchData.hpp"

namespace shamrocl::amr {

    template<class Tcoord, u32 dim>
    class AMRHandler {

        public:
        using CellCoord = AMRCellCoord<Tcoord, dim>;
        static constexpr u32 split_count = CellCoord::splts_count;

        inline static void
        split_cells_all_fields(shamrock::patch::PatchData &pdat, sycl::buffer<u32> &split_idx) {




        }
    };

} // namespace shamrocl::amr