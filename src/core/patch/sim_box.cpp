#include "sim_box.hpp"

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