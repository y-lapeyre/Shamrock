// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Model.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

#include <random>

namespace shamphys {

    /**
     * @brief Utility to generate SPH particle positions for an axisymetric disc described by a 
     * radial surface density profile sigma(r) and a vertical gaussian profile with std mean H(r)
     * 
     * @tparam Tvec 
     */
    template<class Tvec>
    class SPHDiscAxysimetricVerticalGaussianGenerator {
        public:
        using Tscal = shambase::VecComponent<Tvec>;

        class Iterator {
            bool done = false;
            u64 Npart;
            u64 current_index;

            std::mt19937 eng;

            // for next_r
            Tscal r_in;
            Tscal r_out;
            std::function<Tscal(Tscal)> sigma_profile;

            // for next_position
            Tvec center;
            std::function<Tscal(Tscal)> H_profile;

            Tvec ex = {1, 0, 0};
            Tvec ey = {0, 1, 0};
            Tvec ez = {0, 0, 1};

            public:
            Iterator() {

                if (Npart == 0) {
                    done = true;
                }
            }

            inline bool is_done() { return done; }

            /**
             * @brief get next radial position assuming input sigma profile
             *
             * @return Tscal
             */
            inline Tscal next_r() {
                constexpr Tscal _2pi = 2 * shambase::constants::pi<Tscal>;

                auto f_func = [&](Tscal r) {
                    return r * sigma_profile(r);
                }; // adjust density proba func to a radial profile being sigma

                Tscal fmax = f_func(r_out); // check if this is the real max

                auto find_r = [&]() {
                    while (true) {
                        Tscal u2 = shamalgs::random::mock_value<Tscal>(eng, 0, fmax);
                        Tscal r  = shamalgs::random::mock_value<Tscal>(eng, r_in, r_out);
                        if (u2 < f_func(r)) {
                            return r;
                        }
                    }
                };
                return find_r();
            }

            inline Tvec next_position() {

                Tscal r = next_r();

                auto theta = shamalgs::random::mock_value<Tscal>(eng, 0, _2pi);

                auto Gauss = shamalgs::random::mock_gaussian<Tscal>(eng);

                Tscal z = H_profile(r) * Gauss;

                auto pos = ex * r * sycl::cos(theta) + ey * r * sycl::sin(theta) + ez * z;

                return pos + center;
            }

            inline std::vector<Tvec> next_n_position(u32 nmax) {
                std::vector<Tvec> ret{};
                for (u32 i = 0; i < nmax; i++) {
                    if (done) {
                        break;
                    }

                    ret.push_back(next_position());
                }
                return ret;
            }
        };
    };

} // namespace shamphys