// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "CoordRangeTransform.hpp"

#include "shamsys/legacy/log.hpp"

namespace shammath {

    template<class Tsource, class Tdest>
    void CoordRangeTransform<Tsource, Tdest>::print_transform() {
        if (mode == multiply) {
            logger::debug_ln(
                "CoordRangeTransform", "multiply:", fact, source_coord_min, dest_coord_min
            );
        } else {
            logger::debug_ln(
                "CoordRangeTransform", "divide  :", fact, source_coord_min, dest_coord_min
            );
        }
    }

    template<>
    CoordRangeTransform<u64_3, f32_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<f32_3> dest_range
    )
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        mode = multiply;

        u64_3 source_delt = source_range.delt();
        f32_3 dest_delt   = dest_range.delt();

        f32_3 source_sz_conv = source_delt.convert<f32>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u64_3, f64_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<f64_3> dest_range
    )
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        mode = multiply;

        u64_3 source_delt = source_range.delt();
        f64_3 dest_delt   = dest_range.delt();

        f64_3 source_sz_conv = source_delt.convert<f64>();

        fact = dest_delt / source_sz_conv;
    }

    template<class T>
    void check_divisor_throw(T val, T divisor);

    template<>
    void check_divisor_throw(u64_3 val, u64_3 divisor) {
        bool cmp_x = val.x() % divisor.x() == 0;
        bool cmp_y = val.y() % divisor.y() == 0;
        bool cmp_z = val.z() % divisor.z() == 0;

        if (!cmp_x) {
            throw excep_with_pos(
                std::invalid_argument,
                "the divisor does not divide the value on component x\n"
                "  val     = (" +
                    std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", " +
                    std::to_string(val.z()) + ")\n" + "  divisor = (" +
                    std::to_string(divisor.x()) + ", " + std::to_string(divisor.y()) + ", " +
                    std::to_string(divisor.z()) + ")\n"
            );
        }

        if (!cmp_y) {
            throw excep_with_pos(
                std::invalid_argument,
                "the divisor does not divide the value on component y\n"
                "  val     = (" +
                    std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", " +
                    std::to_string(val.z()) + ")\n" + "  divisor = (" +
                    std::to_string(divisor.x()) + ", " + std::to_string(divisor.y()) + ", " +
                    std::to_string(divisor.z()) + ")\n"
            );
        }

        if (!cmp_z) {
            throw excep_with_pos(
                std::invalid_argument,
                "the divisor does not divide the value on component z\n"
                "  val     = (" +
                    std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", " +
                    std::to_string(val.z()) + ")\n" + "  divisor = (" +
                    std::to_string(divisor.x()) + ", " + std::to_string(divisor.y()) + ", " +
                    std::to_string(divisor.z()) + ")\n"
            );
        }
    }

    template<>
    CoordRangeTransform<u64_3, u64_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<u64_3> dest_range
    )
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        u64_3 source_delt = source_range.delt();
        u64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = dest_delt / source_delt;
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = source_delt / dest_delt;
            }

        } else {
            throw excep_with_pos(std::invalid_argument, "the range comparaison are not the same");
        }

        print_transform();
    }

    template<>
    CoordRangeTransform<u64_3, u32_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<u32_3> dest_range
    )
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        throw excep_with_pos(std::invalid_argument, "not implemented");
    }

    template class CoordRangeTransform<u64_3, f32_3>;
    template class CoordRangeTransform<u64_3, f64_3>;
    template class CoordRangeTransform<u64_3, u32_3>;
    template class CoordRangeTransform<u64_3, u64_3>;

} // namespace shammath
