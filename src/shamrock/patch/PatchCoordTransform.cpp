// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "PatchCoordTransform.hpp"
namespace shamrock::patch {

    template<>
    PatchCoordTransform<f32_3>::PatchCoordTransform(
        u64_3 pcoord_min, u64_3 pcoord_max, f32_3 obj_coord_min, f32_3 obj_coord_max
    ) {
        mode = multiply;

        u64_3 patch_delt = pcoord_max - pcoord_min;
        f32_3 obj_delt   = obj_coord_max - obj_coord_min;

        f32_3 patch_b_size = patch_delt.convert<f32>();

        fact = obj_delt / patch_b_size;
    }

    template<>
    PatchCoordTransform<f64_3>::PatchCoordTransform(
        u64_3 pcoord_min, u64_3 pcoord_max, f64_3 obj_coord_min, f64_3 obj_coord_max
    ) {
        mode = multiply;

        u64_3 patch_delt = pcoord_max - pcoord_min;
        f64_3 obj_delt   = obj_coord_max - obj_coord_min;

        f64_3 patch_b_size = patch_delt.convert<f64>();

        fact = obj_delt / patch_b_size;
    }

    template<>
    PatchCoordTransform<u64_3>::PatchCoordTransform(
        u64_3 pcoord_min, u64_3 pcoord_max, u64_3 obj_coord_min, u64_3 obj_coord_max
    ) {
        u64_3 patch_delt = pcoord_max - pcoord_min;
        u64_3 obj_delt   = obj_coord_max - obj_coord_min;

        u64_3 patch_b_size = patch_delt.convert<u64>();

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

        return;
    }

    template<>
    PatchCoordTransform<u32_3>::PatchCoordTransform(
        u64_3 pcoord_min, u64_3 pcoord_max, u32_3 obj_coord_min, u32_3 obj_coord_max
    ) {
        u64_3 patch_delt = pcoord_max - pcoord_min;
        u32_3 obj_delt   = obj_coord_max - obj_coord_min;

        //TODO check no overflow

        u32_3 patch_b_size = patch_delt.convert<u32>();

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

        return;
    }

} // namespace shamrock::patch