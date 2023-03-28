// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shammath/CoordRange.hpp"
#include "shambase/sycl.hpp"

namespace shamrock::patch {

    class PatchCoord {
        public:
        static constexpr u32 dim         = 3U;
        static constexpr u32 splts_count = 1U << dim;

        u64 x_min, y_min, z_min;
        u64 x_max, y_max, z_max;

        PatchCoord() = default;

        PatchCoord(u64 x_min, u64 y_min, u64 z_min, u64 x_max, u64 y_max, u64 z_max)
            : x_min(x_min), y_min(y_min), z_min(z_min), x_max(x_max), y_max(y_max), z_max(z_max) {}

        [[nodiscard]] inline static auto
        get_split_coord(u64 x_min, u64 y_min, u64 z_min, u64 x_max, u64 y_max, u64 z_max)
            -> std::array<u64, dim> {
            return {
                (((x_max - x_min) + 1) / 2) - 1 + x_min,
                (((y_max - y_min) + 1) / 2) - 1 + y_min,
                (((z_max - z_min) + 1) / 2) - 1 + z_min};
        }


        inline static auto
        get_split(u64 x_min, u64 y_min, u64 z_min, u64 x_max, u64 y_max, u64 z_max)
            -> std::array<PatchCoord, splts_count> {

            PatchCoord p0, p1, p2, p3, p4, p5, p6, p7;

            auto splts = get_split_coord(x_min, y_min, z_min, x_max, y_max, z_max);

            u64 split_x = splts[0];
            u64 split_y = splts[1];
            u64 split_z = splts[2];

            p0.x_min = x_min;
            p0.y_min = y_min;
            p0.z_min = z_min;
            p0.x_max = split_x;
            p0.y_max = split_y;
            p0.z_max = split_z;

            p1.x_min = x_min;
            p1.y_min = y_min;
            p1.z_min = split_z + 1;
            p1.x_max = split_x;
            p1.y_max = split_y;
            p1.z_max = z_max;

            p2.x_min = x_min;
            p2.y_min = split_y + 1;
            p2.z_min = z_min;
            p2.x_max = split_x;
            p2.y_max = y_max;
            p2.z_max = split_z;

            p3.x_min = x_min;
            p3.y_min = split_y + 1;
            p3.z_min = split_z + 1;
            p3.x_max = split_x;
            p3.y_max = y_max;
            p3.z_max = z_max;

            p4.x_min = split_x + 1;
            p4.y_min = y_min;
            p4.z_min = z_min;
            p4.x_max = x_max;
            p4.y_max = split_y;
            p4.z_max = split_z;

            p5.x_min = split_x + 1;
            p5.y_min = y_min;
            p5.z_min = split_z + 1;
            p5.x_max = x_max;
            p5.y_max = split_y;
            p5.z_max = z_max;

            p6.x_min = split_x + 1;
            p6.y_min = split_y + 1;
            p6.z_min = z_min;
            p6.x_max = x_max;
            p6.y_max = y_max;
            p6.z_max = split_z;

            p7.x_min = split_x + 1;
            p7.y_min = split_y + 1;
            p7.z_min = split_z + 1;
            p7.x_max = x_max;
            p7.y_max = y_max;
            p7.z_max = z_max;

            return {p0, p1, p2, p3, p4, p5, p6, p7};
        }

        inline auto split() -> std::array<PatchCoord, splts_count> {
            return get_split(x_min, y_min, z_min, x_max, y_max, z_max);
        }

        inline static PatchCoord merge(PatchCoord c1, PatchCoord c2) {
            return PatchCoord(
                sycl::min(c1.x_min, c2.x_min),
                sycl::min(c1.y_min, c2.y_min),
                sycl::min(c1.z_min, c2.z_min),
                sycl::max(c1.x_max, c2.x_max),
                sycl::max(c1.y_max, c2.y_max),
                sycl::max(c1.z_max, c2.z_max)
            );
        }

        inline static PatchCoord merge(std::array<PatchCoord, splts_count> others) {
            return merge(
                merge(merge(others[0], others[1]), merge(others[2], others[3])),
                merge(merge(others[4], others[5]), merge(others[6], others[7]))
            );
        }

        shammath::CoordRange<u64_3> get_patch_range() const {
            return {
                u64_3{x_min,y_min,z_min},
                u64_3{x_max,y_max,z_max} + 1 
            };
        }

        template<class T>
        inline static std::tuple<sycl::vec<T, 3>, sycl::vec<T, 3>> convert_coord(
            u64 x_min,
            u64 y_min,
            u64 z_min,
            u64 x_max,
            u64 y_max,
            u64 z_max,

            u64 offset_x,
            u64 offset_y,
            u64 offset_z,

            sycl::vec<T, 3> divfact,
            sycl::vec<T, 3> offset
        ) {

            using vec = sycl::vec<T, 3>;

            vec min_bound =
                vec{x_min - offset_x, y_min - offset_y, z_min - offset_z} / divfact + offset;
            vec max_bound =
                (vec{x_max - offset_x, y_max - offset_y, z_max - offset_z} + 1) / divfact + offset;

            return {min_bound, max_bound};
        }
    };
} // namespace shamrock::patch