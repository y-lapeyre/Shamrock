// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "CoordRange.hpp"

namespace shammath {

    template<class Tsource, class Tdest>
    class CoordRangeTransform {

        using SourceProp = shamutils::sycl_utils::VectorProperties<Tsource>;
        using DestProp   = shamutils::sycl_utils::VectorProperties<Tdest>;

        using component_source_t = typename SourceProp::component_type;
        using component_dest_t   = typename DestProp::component_type;

        enum TransformFactMode { multiply, divide };

        TransformFactMode mode;

        // written as Patch->Coord transform
        Tdest fact;

        Tdest dest_coord_min;
        Tsource source_coord_min;

        public:
        static_assert(
            SourceProp::dimension == DestProp::dimension,
            "input and output dimensions should be the same"
        );

        CoordRangeTransform(CoordRange<Tsource> source_range, CoordRange<Tdest> dest_range) {}

        CoordRange<Tdest> transform(CoordRange<Tsource> rnge);
        CoordRange<Tsource> reverse_transform(CoordRange<Tdest> rnge);

        void print_transform();
    };

    //////////////////////////////////
    // out of line impl
    //////////////////////////////////

    template<class Tsource, class Tdest>
    inline CoordRange<Tdest>
    CoordRangeTransform<Tsource, Tdest>::transform(CoordRange<Tsource> rnge) {

        u64_3 pmin = rnge.lower;
        u64_3 pmax = rnge.upper;

        if (mode == multiply) {
            return {
                ((pmin - source_coord_min).template convert<component_dest_t>()) * fact +
                    dest_coord_min,
                ((pmax - source_coord_min).template convert<component_dest_t>()) * fact +
                    dest_coord_min};
        } else {
            return {
                ((pmin - source_coord_min).template convert<component_dest_t>()) / fact +
                    dest_coord_min,
                ((pmax - source_coord_min).template convert<component_dest_t>()) / fact +
                    dest_coord_min};
        }
    }

    template<class Tsource, class Tdest>
    inline CoordRange<Tsource>
    CoordRangeTransform<Tsource, Tdest>::reverse_transform(CoordRange<Tdest> rnge) {

        u64_3 pmin;
        u64_3 pmax;

        if (mode == multiply) {
            return {
                ((rnge.lower - dest_coord_min) / fact).template convert<component_source_t>() +
                    source_coord_min,
                ((rnge.upper - dest_coord_min) / fact).template convert<component_source_t>() +
                    source_coord_min};
        } else {
            return {
                ((rnge.lower - dest_coord_min) * fact).template convert<component_source_t>() +
                    source_coord_min,
                ((rnge.upper - dest_coord_min) * fact).template convert<component_source_t>() +
                    source_coord_min};
        }
    }

} // namespace shammath