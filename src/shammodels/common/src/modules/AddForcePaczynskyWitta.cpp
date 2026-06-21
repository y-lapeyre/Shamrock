// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddForcePaczynskyWitta.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/common/modules/AddForcePaczynskyWitta.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::common::modules {

    template<class Tvec>
    void AddForcePaczynskyWitta<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        edges.spans_positions.check_sizes(edges.sizes.indexes);
        edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

        Tscal cmass = edges.central_mass.data;
        Tscal G     = edges.constant_G.data;
        Tscal c     = edges.constant_c.data;
        Tvec cpos   = edges.central_pos.data;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_positions.get_spans()},
            sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
            edges.sizes.indexes,
            [GM = -cmass * G, c, cpos](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                Tscal rs            = 2 * GM / (c * c);
                Tvec r_a            = xyz[gid] - cpos;
                Tscal abs_ra        = sycl::length(r_a);
                Tscal abs_ra_m_rs_3 = (abs_ra - rs) * (abs_ra - rs) * (abs_ra - rs);

                axyz_ext[gid] += GM * r_a / abs_ra_m_rs_3;
            });
    }

    template<class Tvec>
    inline std::string AddForcePaczynskyWitta<Tvec>::_impl_get_tex() const {

        auto constant_G   = get_ro_edge_base(0).get_tex_symbol();
        auto constant_c   = get_ro_edge_base(1).get_tex_symbol();
        auto central_mass = get_ro_edge_base(2).get_tex_symbol();
        auto central_pos  = get_ro_edge_base(3).get_tex_symbol();
        auto positions    = get_ro_edge_base(4).get_tex_symbol();
        auto axyz_ext     = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
                 Add force (Paczynski-Witta potential)

                 \begin{align}
                 r_{\text{s}} &= 2 * {constant_G}* {central_mass} / {constant_c}^2\\
                 {axyz_ext}_i &= -{constant_G} * {central_mass} * {central_pos}_i / ({positions}_i - r_{\text{s}})^3
                 \end{align}
             )tex";

        shambase::replace_all(tex, "{constant_G}", constant_G);
        shambase::replace_all(tex, "{central_mass}", central_mass);
        shambase::replace_all(tex, "{central_pos}", central_pos);
        shambase::replace_all(tex, "{positions}", positions);
        shambase::replace_all(tex, "{axyz_ext}", axyz_ext);

        return tex;
    }

    template class shammodels::common::modules::AddForcePaczynskyWitta<f64_3>;

} // namespace shammodels::common::modules
