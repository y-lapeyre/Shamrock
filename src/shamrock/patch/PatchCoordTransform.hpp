// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamutils/sycl_utils/vectorProperties.hpp"
#include "shammath/CoordRange.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchCoord.hpp"
#include "shamsys/legacy/log.hpp"
#include <stdexcept>
#include <type_traits>
#include <types.hpp>

namespace shamrock::patch {

    template<class Tcoord>
    class PatchCoordTransform {

        using CoordProp = shamutils::sycl_utils::VectorProperties<Tcoord>;

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

        inline PatchCoordTransform(shammath::CoordRange<u64_3> patch_range, shammath::CoordRange<Tcoord> obj_range)
            : PatchCoordTransform(
                  patch_range.lower,
                  patch_range.upper,
                  obj_range.lower,
                  obj_range.upper
              ) {}

        
        shammath::CoordRange<Tcoord> to_obj_coord(shammath::CoordRange<u64_3> p);
        PatchCoord to_patch_coord(shammath::CoordRange<Tcoord> obj);

        inline shammath::CoordRange<Tcoord> to_obj_coord(Patch p){
            return to_obj_coord(p.get_patch_range());
        }

        inline void print_transform(){
            if(mode == multiply){
                logger::debug_ln("PathCoordTransform","multiply:",fact ,obj_coord_min,patch_coord_min);
            }else{
                logger::debug_ln("PathCoordTransform","divide  :",fact ,obj_coord_min,patch_coord_min);
            }
            
        }
    };

    //////////////////////////////////
    // out of line impl
    //////////////////////////////////


    template<class Tcoord>
    inline shammath::CoordRange<Tcoord> PatchCoordTransform<Tcoord>::to_obj_coord(shammath::CoordRange<u64_3> p) {

        u64_3 pmin = p.lower;
        u64_3 pmax = p.upper;

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
    inline PatchCoord PatchCoordTransform<Tcoord>::to_patch_coord(shammath::CoordRange<Tcoord> c) {

        u64_3 pmin;
        u64_3 pmax;

        if (mode == multiply) {
            return {
                ((c.min - obj_coord_min) / fact).template convert<u64>() + patch_coord_min,
                ((c.upper - obj_coord_min) / fact).template convert<u64>() + patch_coord_min};
        } else {
            return {
                ((c.min - obj_coord_min) * fact).template convert<u64>() + patch_coord_min,
                ((c.upper - obj_coord_min) * fact).template convert<u64>() + patch_coord_min};
        }
    }

} // namespace shamrock::patch