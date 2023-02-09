// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

namespace shamrocl::amr {

    template<class Tcoord, u32 dim>
    class AMRCellCoord {

        public:
        static constexpr u32 splts_count = 1U << dim;
        Tcoord bmin, bmax;

        [[nodiscard]] inline static auto get_split_coord(Tcoord bmin, Tcoord bmax) -> Tcoord {
            return (bmax - bmin) / 2 + bmin;
        }

        inline static auto get_split(Tcoord bmin, Tcoord bmax)
            -> std::array<AMRCellCoord, splts_count> {

            AMRCellCoord p0, p1, p2, p3, p4, p5, p6, p7;

            Tcoord splts = get_split_coord(bmin, bmax);

            p0.bmin.x() = bmin.x();
            p0.bmin.y() = bmin.y();
            p0.bmin.z() = bmin.z();
            p0.bmax.x() = splts.x();
            p0.bmax.y() = splts.y();
            p0.bmax.z() = splts.z();

            p1.bmin.x() = bmin.x();
            p1.bmin.y() = bmin.y();
            p1.bmin.z() = splts.z();
            p1.bmax.x() = splts.x();
            p1.bmax.y() = splts.y();
            p1.bmax.z() = bmax.z();

            p2.bmin.x() = bmin.x();
            p2.bmin.y() = splts.y();
            p2.bmin.z() = bmin.z();
            p2.bmax.x() = splts.x();
            p2.bmax.y() = bmax.y();
            p2.bmax.z() = splts.z();

            p3.bmin.x() = bmin.x();
            p3.bmin.y() = splts.y();
            p3.bmin.z() = splts.z();
            p3.bmax.x() = splts.x();
            p3.bmax.y() = bmax.y();
            p3.bmax.z() = bmax.z();

            p4.bmin.x() = splts.x();
            p4.bmin.y() = bmin.y();
            p4.bmin.z() = bmin.z();
            p4.bmax.x() = bmax.x();
            p4.bmax.y() = splts.y();
            p4.bmax.z() = splts.z();

            p5.bmin.x() = splts.x();
            p5.bmin.y() = bmin.y();
            p5.bmin.z() = splts.z();
            p5.bmax.x() = bmax.x();
            p5.bmax.y() = splts.y();
            p5.bmax.z() = bmax.z();

            p6.bmin.x() = splts.x();
            p6.bmin.y() = splts.y();
            p6.bmin.z() = bmin.z();
            p6.bmax.x() = bmax.x();
            p6.bmax.y() = bmax.y();
            p6.bmax.z() = splts.z();

            p7.bmin.x() = splts.x();
            p7.bmin.y() = splts.y();
            p7.bmin.z() = splts.z();
            p7.bmax.x() = bmax.x();
            p7.bmax.y() = bmax.y();
            p7.bmax.z() = bmax.z();

            return {p0, p1, p2, p3, p4, p5, p6, p7};
        }
    };

} // namespace shamrocl::amr