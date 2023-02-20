// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shammath/sycl_utils/vectorProperties.hpp"
#include "shamrock/math/CoordRange.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchCoord.hpp"
#include <stdexcept>
#include <type_traits>
#include <types.hpp>

namespace shamrock::patch {

    template<class Tcoord>
    class PatchCoordTransform {

        using CoordProp = shammath::sycl_utils::VectorProperties<Tcoord>;

        enum TransformFactMode { multiply, divide };

        TransformFactMode mode;

        // written as Patch->Coord transform
        Tcoord fact;

        Tcoord obj_coord_min;
        u64_3 patch_coord_min;

        public:
        PatchCoordTransform(
            u64_3 pcoord_min, u64_3 pcoord_max, Tcoord obj_coord_min, Tcoord obj_coord_max
        );

        inline PatchCoordTransform(CoordRange<u64_3> patch_range, CoordRange<Tcoord> obj_range)
            : PatchCoordTransform(
                  patch_range.low_bound,
                  patch_range.high_bound,
                  obj_range.low_bound,
                  obj_range.high_bound
              ) {}

        
        CoordRange<Tcoord> to_obj_coord(CoordRange<u64_3> p);
        PatchCoord to_patch_coord(CoordRange<Tcoord> obj);

        inline CoordRange<Tcoord> to_obj_coord(Patch p){
            return to_obj_coord(p.get_patch_range());
        }
    };

    //////////////////////////////////
    // out of line impl
    //////////////////////////////////


    template<class Tcoord>
    inline CoordRange<Tcoord> PatchCoordTransform<Tcoord>::to_obj_coord(CoordRange<u64_3> p) {

        u64_3 pmin = p.low_bound;
        u64_3 pmax = p.high_bound;

        if (mode == multiply) {
            return {
                ((pmin - patch_coord_min).convert<typename CoordProp::component_type>()) * fact +
                    obj_coord_min,
                ((pmax - patch_coord_min).convert<typename CoordProp::component_type>()) * fact +
                    obj_coord_min};
        } else {
            return {
                ((pmin - patch_coord_min).convert<typename CoordProp::component_type>()) / fact +
                    obj_coord_min,
                ((pmax - patch_coord_min).convert<typename CoordProp::component_type>()) / fact +
                    obj_coord_min};
        }
    }

    template<class Tcoord>
    inline PatchCoord PatchCoordTransform<Tcoord>::to_patch_coord(CoordRange<Tcoord> c) {

        u64_3 pmin;
        u64_3 pmax;

        if (mode == multiply) {
            return {
                ((c.low_bound - obj_coord_min) / fact).template convert<u64>() + patch_coord_min,
                ((c.high_bound - obj_coord_min) / fact).template convert<u64>() + patch_coord_min};
        } else {
            return {
                ((c.low_bound - obj_coord_min) * fact).template convert<u64>() + patch_coord_min,
                ((c.high_bound - obj_coord_min) * fact).template convert<u64>() + patch_coord_min};
        }
    }

} // namespace shamrock::patch