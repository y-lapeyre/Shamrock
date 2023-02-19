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

        CoordRange<Tcoord> to_obj_coord(Patch p);
        PatchCoord to_patch_coord(CoordRange<Tcoord> p);
    };

    //////////////////////////////////
    // out of line impl
    //////////////////////////////////

    template<class Tcoord>
    inline PatchCoordTransform<Tcoord>::PatchCoordTransform(
        u64_3 pcoord_min, u64_3 pcoord_max, Tcoord obj_coord_min, Tcoord obj_coord_max
    )
        : obj_coord_min(obj_coord_min), patch_coord_min(pcoord_min) {

        if constexpr (CoordProp::dimension == 3) {
            if constexpr (CoordProp::is_float_based) {

                mode = multiply;

                u64_3 patch_delt = pcoord_max - pcoord_min;
                Tcoord obj_delt  = obj_coord_max - obj_coord_min;

                Tcoord patch_b_size = patch_delt.convert<typename CoordProp::component_type>();

                fact = obj_delt / patch_b_size;
            }

            if constexpr (CoordProp::is_uint_based) {

                u64_3 patch_delt = pcoord_max - pcoord_min;
                Tcoord obj_delt  = obj_coord_max - obj_coord_min;

                if constexpr (std::is_same<typename CoordProp::component_type, u64>::value) {
                    Tcoord patch_b_size = patch_delt.convert<typename CoordProp::component_type>();

                    bool cmp_x = obj_delt.x() >= patch_b_size.x();
                    bool cmp_y = obj_delt.y() >= patch_b_size.y();
                    bool cmp_z = obj_delt.z() >= patch_b_size.z();

                    if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

                        bool obj_greater_than_patch = cmp_x;

                        if (obj_greater_than_patch) {
                            mode = multiply;
                            fact = obj_delt / patch_b_size;
                        } else {
                            mode = divide;
                            fact = patch_b_size / obj_delt;
                        }

                    } else {
                        throw std::invalid_argument("the range comparaison are not the same");
                    }
                }
            }
        }

        throw std::invalid_argument("the current case is not handled");
    }

    template<class Tcoord>
    inline CoordRange<Tcoord> PatchCoordTransform<Tcoord>::to_obj_coord(Patch p) {

        u64_3 pmin{p.x_min, p.y_min, p.z_min};
        u64_3 pmax{p.x_max, p.y_max, p.z_max};

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