// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file crystalLattice.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/aliases_int.hpp"
#include "shambase/sycl_utils.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammath/CoordRange.hpp"
#include <array>
#include <functional>
#include <utility>
#include <vector>

namespace shammath {

    class LatticeError : public std::exception {
        public:
        explicit LatticeError(const char *message) : msg_(message) {}

        explicit LatticeError(const std::string &message) : msg_(message) {}

        ~LatticeError() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

        protected:
        std::string msg_;
    };

    template<class Tvec>
    class LatticeHCP {
        public:
        static constexpr u32 dim = 3;

        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @brief generate a HCP lattice centered on (0,0,0)
         *
         * @param dr
         * @param coord
         * @return constexpr Tvec
         */
        static inline constexpr Tvec generator(Tscal dr, std::array<i32, dim> coord) noexcept {

            i32 i = coord[0];
            i32 j = coord[1];
            i32 k = coord[2];

            Tvec r_a = {
                2 * i + ((j + k) % 2),
                sycl::sqrt(3.) * (j + (1. / 3.) * (k % 2)),
                2 * sycl::sqrt(6.) * k / 3};

            return dr * r_a;
        }

        static inline constexpr CoordRange<Tvec>
        get_periodic_box(Tscal dr, std::array<i32, dim> coord_min, std::array<i32, dim> coord_max) {
            Tscal xmin, xmax, ymin, ymax, zmin, zmax;

            xmin = 2 * coord_min[0];
            xmax = 2 * coord_max[0];

            ymin = sycl::sqrt(3.) * coord_min[1];
            ymax = sycl::sqrt(3.) * coord_max[1];

            zmin = 2 * sycl::sqrt(6.) * coord_min[2] / 3;
            zmax = 2 * sycl::sqrt(6.) * coord_max[2] / 3;

            if (coord_max[0] - coord_min[0] < 2) {
                throw LatticeError("x axis count should be greater than 1");
            }

            if ((coord_max[1] + coord_min[1]) % 2 != 0) {
                throw LatticeError("y axis count should be even");
            }

            if ((coord_max[2] + coord_min[2]) % 2 != 0) {
                throw LatticeError("z axis count should be even");
            }

            return {Tvec{xmin, ymin, zmin} * dr, Tvec{xmax, ymax, zmax} * dr};
        }

        class Iterator {
            Tscal dr;
            std::array<i32, dim> coord_min;
            std::array<i32, dim> coord_max;
            std::array<i32, dim> current;

            using Selector = std::function<bool(Tvec)>;

            Selector selector;
            bool done = false;

            public:
            Iterator(
                Tscal dr,
                std::array<i32, dim> coord_min,
                std::array<i32, dim> coord_max,
                Selector &&selector)
                : dr(dr), coord_min(coord_min), coord_max(coord_max), current(coord_min),
                  selector(std::forward<Selector>(selector)) {}

            bool is_done() { return done; }

            Tvec next() {
                Tvec ret = generator(dr, current);

                current[0]++;
                if (current[0] >= coord_max[0]) {
                    current[0] = coord_min[0];

                    current[1]++;
                    if (current[1] >= coord_max[1]) {
                        current[1] = coord_min[1];

                        current[2]++;
                        if (current[2] >= coord_max[2]) {
                            done = true;
                        }
                    }
                }

                return ret;
            }

            std::vector<Tvec> next_n(u32 nmax) {
                std::vector<Tvec> ret{};
                for (u32 i = 0; i < nmax; i++) {
                    if (done) {
                        break;
                    }

                    ret.push_back(next());
                }
                return ret;
            }
        };
    };

} // namespace shammath