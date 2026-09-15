// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeGravWave.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Compute the gravitational wave quadrupole. Based on Toscani et. al. 2021.
 *
 */

#include "shammodels/common/modules/ComputeGravWave.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {

    namespace {

        // 3x3 mdspan view over a fixed-size array
        template<class Tscal>
        using Mat3 = std::mdspan<Tscal, std::extents<std::size_t, 3, 3>>;

    } // namespace

    template<class Tvec>
    void ComputeGravWave<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        // ------------------------------------------------------------------
        // Size checks
        // ------------------------------------------------------------------
        edges.spans_positions.check_sizes(edges.sizes.indexes);
        edges.spans_velocities.check_sizes(edges.sizes.indexes);
        edges.spans_accelerations.check_sizes(edges.sizes.indexes);
        edges.spans_masses.check_sizes(edges.sizes.indexes);
        edges.spans_accel_ext.check_sizes(edges.sizes.indexes);

        // ------------------------------------------------------------------
        // Unpack scalar inputs
        // ------------------------------------------------------------------
        const Tvec x0       = edges.central_pos.data;
        const Tvec v0       = edges.central_vel.data;
        const Tvec a0       = edges.central_acc.data;
        const Tscal fac     = edges.gw_prefactor.data;
        const Tscal theta_d = edges.theta_gw.data;
        const Tscal phi_d   = edges.phi_gw.data;

        constexpr Tscal pi = static_cast<Tscal>(3.14159265358979323846264338327950288L);

        // ------------------------------------------------------------------
        // Step 1 : accumulate the six independent components of d²Q/dt²
        //          over all particles of this rank, then MPI-reduce.
        //
        //          Note: the reference Fortran code excludes accreted
        //          particles with xyzh(4,i) <= tiny. We rely on the
        //          caller's mass span already reflecting that (m <= 0 =>
        //          skip). If you also need to keep the accreted mask
        //          explicitly, add an extra span here.
        // ------------------------------------------------------------------
        std::array<Tscal, 6> ddq = sham::distributed_data_kernel_call_reduce<6>(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_positions.get_spans(),
                edges.spans_velocities.get_spans(),
                edges.spans_accelerations.get_spans(),
                edges.spans_masses.get_spans(),
                edges.spans_accel_ext.get_spans()},
            edges.sizes.indexes,
            [x0, v0, a0](
                u32 gid,
                const Tvec *xyz,
                const Tvec *vxyz,
                const Tvec *axyz,
                const Tscal *mass,
                const Tvec *axyz_ext) -> std::array<Tscal, 6> {
                std::array<Tscal, 6> local
                    = {Tscal(0), Tscal(0), Tscal(0), Tscal(0), Tscal(0), Tscal(0)};

                const Tscal m = mass[gid];
                if (m <= Tscal(0)) {
                    return local; // accreted / removed particle
                }

                const Tscal x  = xyz[gid][0] - x0[0];
                const Tscal y  = xyz[gid][1] - x0[1];
                const Tscal z  = xyz[gid][2] - x0[2];
                const Tscal vx = vxyz[gid][0] - v0[0];
                const Tscal vy = vxyz[gid][1] - v0[1];
                const Tscal vz = vxyz[gid][2] - v0[2];

                // acceleration = (total acceleration) - a0  +  external
                const Tscal ax = axyz[gid][0] - a0[0] + axyz_ext[gid][0];
                const Tscal ay = axyz[gid][1] - a0[1] + axyz_ext[gid][1];
                const Tscal az = axyz[gid][2] - a0[2] + axyz_ext[gid][2];

                local[0] = m * (Tscal(2) * vx * vx + x * ax + x * ax); // ddqxx
                local[1] = m * (Tscal(2) * vx * vy + x * ay + y * ax); // ddqxy
                local[2] = m * (Tscal(2) * vx * vz + x * az + z * ax); // ddqxz
                local[3] = m * (Tscal(2) * vy * vy + y * ay + y * ay); // ddqyy
                local[4] = m * (Tscal(2) * vy * vz + y * az + z * ay); // ddqyz
                local[5] = m * (Tscal(2) * vz * vz + z * az + z * az); // ddqzz

                return local;
            });

        // ------------------------------------------------------------------
        // Step 2 : build the symmetric 3x3 quadrupole second-derivative
        //          matrix Q. (Kept full; the reference code does not
        //          subtract the trace either.)
        // ------------------------------------------------------------------
        std::array<Tscal, 9> Q_arr{};
        Mat3<Tscal> Q(Q_arr.data());
        Q(0, 0) = ddq[0];
        Q(0, 1) = ddq[1];
        Q(0, 2) = ddq[2];
        Q(1, 0) = ddq[1];
        Q(1, 1) = ddq[3];
        Q(1, 2) = ddq[4];
        Q(2, 0) = ddq[2];
        Q(2, 1) = ddq[4];
        Q(2, 2) = ddq[5];

        // ------------------------------------------------------------------
        // Step 3 : rotate into the sky plane if theta_gw != 0.
        //
        //   R  = [[ c, 0, s],
        //         [ 0, 1, 0],
        //         [-s, 0, c]]     with c = cos(lambda), s = sin(lambda)
        //
        //   ddq_xy = R^T * Q * R
        // ------------------------------------------------------------------
        std::array<Tscal, 9> ddq_xy_arr{};
        Mat3<Tscal> ddq_xy(ddq_xy_arr.data());

        const bool rotate = std::abs(theta_d) > static_cast<Tscal>(1e-30);
        if (rotate) {
            const Tscal lambda = theta_d * pi / static_cast<Tscal>(180);
            const Tscal c      = std::cos(lambda);
            const Tscal s      = std::sin(lambda);

            std::array<Tscal, 9> R_arr
                = {c, Tscal(0), s, Tscal(0), Tscal(1), Tscal(0), -s, Tscal(0), c};
            Mat3<Tscal> R(R_arr.data());

            // inter = Q * R
            std::array<Tscal, 9> inter_arr{};
            Mat3<Tscal> inter(inter_arr.data());
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    Tscal sum = Tscal(0);
                    for (std::size_t k = 0; k < 3; ++k) {
                        sum += Q(i, k) * R(k, j);
                    }
                    inter(i, j) = sum;
                }
            }

            // ddq_xy = R^T * inter
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    Tscal sum = Tscal(0);
                    for (std::size_t k = 0; k < 3; ++k) {
                        sum += R(k, i) * inter(k, j);
                    }
                    ddq_xy(i, j) = sum;
                }
            }
        } else {
            for (std::size_t i = 0; i < 9; ++i) {
                ddq_xy_arr[i] = Q_arr[i];
            }
        }

        // ------------------------------------------------------------------
        // Step 4 : angular pattern of h+ and hx at the four standard
        //          viewing angles eta = 0, pi/6, pi/3, pi/2.
        // ------------------------------------------------------------------
        const Tscal phi     = phi_d * pi / static_cast<Tscal>(180);
        const Tscal sinphi  = std::sin(phi);
        const Tscal cosphi  = std::cos(phi);
        const Tscal sinphi2 = sinphi * sinphi;
        const Tscal cosphi2 = cosphi * cosphi;
        const Tscal sin2phi = std::sin(Tscal(2) * phi);
        const Tscal cos2phi = std::cos(Tscal(2) * phi);

        std::array<Tscal, 4> hx_out{};
        std::array<Tscal, 4> hp_out{};

        for (int i = 0; i < 4; ++i) {
            const Tscal eta     = static_cast<Tscal>(i) * pi / static_cast<Tscal>(6);
            const Tscal sineta  = std::sin(eta);
            const Tscal coseta  = std::cos(eta);
            const Tscal sineta2 = sineta * sineta;
            const Tscal coseta2 = coseta * coseta;
            const Tscal sin2eta = std::sin(Tscal(2) * eta);

            // h+
            hp_out[i] = fac
                        * (ddq_xy(0, 0) * (cosphi2 - sinphi2 * coseta2)
                           + ddq_xy(1, 1) * (sinphi2 - cosphi2 * coseta2) - ddq_xy(2, 2) * sineta2
                           - ddq_xy(0, 1) * sin2phi * (Tscal(1) + coseta2)
                           + ddq_xy(0, 2) * sinphi * sin2eta + ddq_xy(1, 2) * cosphi * sin2eta);

            // hx
            hx_out[i] = Tscal(2) * fac
                        * (Tscal(0.5) * (ddq_xy(0, 0) - ddq_xy(1, 1)) * sin2phi * coseta
                           + ddq_xy(0, 1) * cos2phi * coseta - ddq_xy(0, 2) * cosphi * sineta
                           + ddq_xy(1, 2) * sinphi * sineta);
        }

        // ------------------------------------------------------------------
        // Step 5 : publish outputs
        // ------------------------------------------------------------------
        edges.ddq.data    = ddq;
        edges.ddq_xy.data = ddq_xy_arr;
        edges.hx.data     = hx_out;
        edges.hp.data     = hp_out;
    }

    template<class Tvec>
    inline std::string ComputeGravWave<Tvec>::_impl_get_tex() const {

        auto positions   = get_ro_edge_base(0).get_tex_symbol();
        auto velocities  = get_ro_edge_base(1).get_tex_symbol();
        auto accels      = get_ro_edge_base(2).get_tex_symbol();
        auto masses      = get_ro_edge_base(3).get_tex_symbol();
        auto accel_ext   = get_ro_edge_base(4).get_tex_symbol();
        auto central_pos = get_ro_edge_base(5).get_tex_symbol();
        auto central_vel = get_ro_edge_base(6).get_tex_symbol();
        auto central_acc = get_ro_edge_base(7).get_tex_symbol();
        auto fac         = get_ro_edge_base(8).get_tex_symbol();
        auto theta       = get_ro_edge_base(9).get_tex_symbol();
        auto phi         = get_ro_edge_base(10).get_tex_symbol();

        auto ddq_out    = get_rw_edge_base(0).get_tex_symbol();
        auto ddq_xy_out = get_rw_edge_base(1).get_tex_symbol();
        auto hx_out     = get_rw_edge_base(2).get_tex_symbol();
        auto hp_out     = get_rw_edge_base(3).get_tex_symbol();

        std::string tex = R"tex(
                Gravitational-wave strain from the quadrupole formula
                (Toscani et al. 2021)

                \begin{align}
                r_i      &= {positions}_i - {central_pos}_i\\
                v_i      &= {velocities}_i - {central_vel}_i\\
                a_i      &= {accels}_i - {central_acc}_i + {accel_ext}_i\\
                \ddot Q_{ij} &= \sum_p {masses}_p
                                \left(2 v_i v_j + r_i a_j + r_j a_i\right)\\
                R        &= \begin{pmatrix}
                              \cos\lambda & 0 & \sin\lambda\\
                              0           & 1 & 0\\
                             -\sin\lambda & 0 & \cos\lambda
                            \end{pmatrix},\quad
                            \lambda = {theta}\,\frac{\pi}{180}\\
                \ddot Q^{xy} &= R^{T}\,\ddot Q\,R\\
                h_+(\eta,\phi) &= {fac}\,
                    \Big[\ddot Q^{xy}_{11}(\cos^2\phi - \sin^2\phi\cos^2\eta)
                        + \ddot Q^{xy}_{22}(\sin^2\phi - \cos^2\phi\cos^2\eta)
                        - \ddot Q^{xy}_{33}\sin^2\eta\\
                    &\qquad - \ddot Q^{xy}_{12}\sin 2\phi\,(1 + \cos^2\eta)
                        + \ddot Q^{xy}_{13}\sin\phi\,\sin 2\eta
                        + \ddot Q^{xy}_{23}\cos\phi\,\sin 2\eta\Big]\\
                h_\times(\eta,\phi) &= 2\,{fac}\,
                    \Big[\tfrac12(\ddot Q^{xy}_{11} - \ddot Q^{xy}_{22})\sin 2\phi\,\cos\eta\\
                    &\qquad + \ddot Q^{xy}_{12}\cos 2\phi\,\cos\eta
                        - \ddot Q^{xy}_{13}\cos\phi\,\sin\eta
                        + \ddot Q^{xy}_{23}\sin\phi\,\sin\eta\Big]
                \end{align}

                Evaluated at $\eta = 0,\ \pi/6,\ \pi/3,\ \pi/2$
                with $\phi = {phi}^\circ$, giving {hx} and {hp}.
            )tex";

        shambase::replace_all(tex, "{positions}", positions);
        shambase::replace_all(tex, "{velocities}", velocities);
        shambase::replace_all(tex, "{accels}", accels);
        shambase::replace_all(tex, "{masses}", masses);
        shambase::replace_all(tex, "{accel_ext}", accel_ext);
        shambase::replace_all(tex, "{central_pos}", central_pos);
        shambase::replace_all(tex, "{central_vel}", central_vel);
        shambase::replace_all(tex, "{central_acc}", central_acc);
        shambase::replace_all(tex, "{fac}", fac);
        shambase::replace_all(tex, "{theta}", theta);
        shambase::replace_all(tex, "{phi}", phi);
        shambase::replace_all(tex, "{hx}", hx_out);
        shambase::replace_all(tex, "{hp}", hp_out);

        (void) ddq_out; // referenced through the outputs already
        (void) ddq_xy_out;

        return tex;
    }

    template class shammodels::common::modules::ComputeGravWave<f64_3>;

} // namespace shammodels::common::modules
