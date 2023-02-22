// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shammath/CoordRange.hpp"
#include "shammath/CoordRangeTransform.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchCoord.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamutils/sycl_utils/vectorProperties.hpp"
#include <stdexcept>
#include <type_traits>
#include <types.hpp>

namespace shamrock::patch {

    template<class Tcoord>
    class PatchCoordTransform {

        using CoordTransform = shammath::CoordRangeTransform<u64_3, Tcoord>;

        CoordTransform transform;

        public:
        inline PatchCoordTransform(
            shammath::CoordRange<u64_3> patch_range, shammath::CoordRange<Tcoord> obj_range
        )
            : transform(patch_range, obj_range) {}

        inline shammath::CoordRange<Tcoord> to_obj_coord(shammath::CoordRange<u64_3> p) {
            return transform.transform(p);
        }

        PatchCoord to_patch_coord(shammath::CoordRange<Tcoord> obj) {
            return transform.reverse_transform(obj);
        }

        inline shammath::CoordRange<Tcoord> to_obj_coord(Patch p) {
            return to_obj_coord(p.get_patch_range());
        }

        inline void print_transform() { transform.print_transform(); }
    };
} // namespace shamrock::patch