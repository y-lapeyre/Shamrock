// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sim_box.hpp"


#if false
template<> 
auto SimulationVolume<f32>::get_patch_volume(Patch &p) -> vec_box{
    vec bmin_p = vec{p.x_min,p.y_min,p.z_min}*scale_factor + translate_factor;
    vec bmax_p = (vec{p.x_max,p.y_max,p.z_max}+ 1)*scale_factor + translate_factor;
    return {bmin_p,bmax_p};
}

template<> 
auto SimulationVolume<f64>::get_patch_volume(Patch &p) -> vec_box{
    vec bmin_p = vec{p.x_min,p.y_min,p.z_min}*scale_factor + translate_factor;
    vec bmax_p = (vec{p.x_max,p.y_max,p.z_max}+ 1)*scale_factor + translate_factor;
    return {bmin_p,bmax_p};
}


template<>
void SimulationVolume<f32>::update_volume(){
    if(bc.type == Free){
        
    }
}
#endif