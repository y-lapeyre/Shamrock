// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/math/vectorManip.hpp"
#include "shamrock/patch/Patch.hpp"
#include "base/patchdata.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
//#include "boundary_condition.hpp"
//#include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/legacy/patch/scheduler/loadbalancing_hilbert.hpp" //TODO remove dependancy from hilbert
#include <memory>
#include <tuple>



#if false
template<class flt>
class SimulationVolume {

    using vec = sycl::vec<flt,3>;

    using vec_box = std::tuple<vec,vec>;

    vec_box box;

    vec_box last_used_volume;

    //computed when update_volume() is called
    vec translate_factor;
    vec scale_factor;

    SimulationDomain<flt> bc;

    public: 

    inline SimulationDomain<flt> & get_boundaries(){
        return bc;
    }

    vec_box get_patch_volume(Patch & p);

    //vec_box apply_boundaries(PatchDataBuffer & p);

    void update_volume();

};
#endif



/**
 * @brief Store the information related to the size of the simulation box to convert patch integer coordinates to floating
 * point ones.
 * //TODO transform this class into boundary condition handler
 */
class SimulationBoxInfo {
  public:

    shamrock::patch::PatchDataLayout & pdl;

    f32_3 min_box_sim_s; ///< minimum coordinate of the box (if single precision)
    f32_3 max_box_sim_s; ///< maximum coordinate of the box (if single precision)

    f64_3 min_box_sim_d; ///< minimum coordinate of the box (if double precision)
    f64_3 max_box_sim_d; ///< maximum coordinate of the box (if double precision)

    // TODO implement box size reduction here

    /**
     * @brief reset box simulation size
     */
    inline void reset_box_size() {

        if(pdl.check_main_field_type<f32_3>()){
            min_box_sim_s = {HUGE_VALF};
            max_box_sim_s = {-HUGE_VALF};
        }else if(pdl.check_main_field_type<f64_3>()){
            min_box_sim_s = {HUGE_VAL};
            max_box_sim_s = {-HUGE_VAL};
        }else{
            throw std::runtime_error(
                __LOC_PREFIX__ + "the chosen type for the main field is not handled"
                );
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
    inline std::tuple<sycl::vec<primtype,3>,sycl::vec<primtype,3>> get_box(shamrock::patch::Patch & p);

    template<>
    inline std::tuple<f32_3,f32_3> get_box<f32>(shamrock::patch::Patch & p){
        using vec3 = sycl::vec<f32,3>;
        using ptype = typename shamrock::math::vec_manip::VectorProperties<vec3>::component_type;

        vec3 translate_factor = min_box_sim_s;
        vec3 div_factor = ptype(HilbertLB::max_box_sz)/(max_box_sim_s - min_box_sim_s);
        return p.convert_coord(div_factor, translate_factor);
    }

    template<>
    inline std::tuple<f64_3,f64_3> get_box<f64>(shamrock::patch::Patch & p){
        using vec3 = sycl::vec<f64,3>;
        using ptype = typename shamrock::math::vec_manip::VectorProperties<vec3>::component_type;
        vec3 translate_factor = min_box_sim_d;
        vec3 div_factor = ptype(HilbertLB::max_box_sz)/(max_box_sim_d - min_box_sim_d);
        return p.convert_coord(div_factor, translate_factor);
        
    }

    inline SimulationBoxInfo(shamrock::patch::PatchDataLayout & pdl) : pdl(pdl){}
};