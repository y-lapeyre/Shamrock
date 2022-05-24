// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_layout.hpp"
#include "patchscheduler/loadbalancing_hilbert.hpp"
#include <tuple>


/**
 * @brief Store the information related to the size of the simulation box to convert patch integer coordinates to floating
 * point ones.
 */
class SimulationBoxInfo {
  public:

    PatchDataLayout & pdl;

    f32_3 min_box_sim_s; ///< minimum coordinate of the box (if single precision)
    f32_3 max_box_sim_s; ///< maximum coordinate of the box (if single precision)

    f64_3 min_box_sim_d; ///< minimum coordinate of the box (if double precision)
    f64_3 max_box_sim_d; ///< maximum coordinate of the box (if double precision)

    // TODO implement box size reduction here

    /**
     * @brief reset box simulation size
     */
    inline void reset_box_size() {

        if (pdl.xyz_mode == xyz32) {
            min_box_sim_s = {HUGE_VALF};
            max_box_sim_s = {-HUGE_VALF};
        }

        if (pdl.xyz_mode == xyz64) {
            min_box_sim_s = {HUGE_VAL};
            max_box_sim_s = {-HUGE_VAL};
        }
    }

    //TODO replace vectype primtype in the code by primtype and sycl::vec<primtype,3> for the others
    template<class primtype>
    void clean_box(primtype tol);

    template<>
    inline void clean_box<f32>(f32 tol){
        f32_3 center = (min_box_sim_s + max_box_sim_s) / 2;
        f32_3 cur_delt = max_box_sim_s - min_box_sim_s;
        cur_delt /= 2;

        cur_delt *= tol;

        min_box_sim_s = center - cur_delt;
        max_box_sim_s = center + cur_delt;
    }

    template<>
    inline void clean_box<f64>(f64 tol){
        f64_3 center = (min_box_sim_d + max_box_sim_d) / 2;
        f64_3 cur_delt = max_box_sim_d - min_box_sim_d;
        cur_delt /= 2;

        cur_delt *= tol;

        min_box_sim_d = center - cur_delt;
        max_box_sim_d = center + cur_delt;
    }

    template<class primtype>
    inline std::tuple<sycl::vec<primtype,3>,sycl::vec<primtype,3>> get_box(Patch & p);

    template<>
    inline std::tuple<f32_3,f32_3> get_box<f32>(Patch & p){
        using vec3 = sycl::vec<f32,3>;
        vec3 translate_factor = min_box_sim_s;
        vec3 scale_factor = (max_box_sim_s - min_box_sim_s)/HilbertLB::max_box_sz;

        vec3 bmin_p = vec3{p.x_min,p.y_min,p.z_min}*scale_factor + translate_factor;
        vec3 bmax_p = (vec3{p.x_max,p.y_max,p.z_max}+ 1)*scale_factor + translate_factor;
        return {bmin_p,bmax_p};
    }

    template<>
    inline std::tuple<f64_3,f64_3> get_box<f64>(Patch & p){
        using vec3 = sycl::vec<f64,3>;
        vec3 translate_factor = min_box_sim_d;
        vec3 scale_factor = (max_box_sim_d - min_box_sim_d)/HilbertLB::max_box_sz;

        vec3 bmin_p = vec3{p.x_min,p.y_min,p.z_min}*scale_factor + translate_factor;
        vec3 bmax_p = (vec3{p.x_max,p.y_max,p.z_max}+ 1)*scale_factor + translate_factor;
        return {bmin_p,bmax_p};
        
    }

    inline SimulationBoxInfo(PatchDataLayout & pdl) : pdl(pdl){}
};