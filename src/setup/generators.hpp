#pragma once

#include "aliases.hpp"

template<class flt,class Tpred_select,class Tpred_pusher>
inline void add_particles_fcc(
    flt r_particle, 
    std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box,
    Tpred_select && selector,
    Tpred_pusher && part_pusher ){
    
    using vec3 = sycl::vec<flt, 3>;

    vec3 box_min = std::get<0>(box);
    vec3 box_max = std::get<1>(box);

    vec3 box_dim = box_max - box_min;

    vec3 iboc_dim = (box_dim / 
        vec3({
            2,
            sycl::sqrt(3.),
            2*sycl::sqrt(6.)/3
        }))/r_particle;

    std::cout << "len vector : (" << iboc_dim.x() << ", " << iboc_dim.y() << ", " << iboc_dim.z() << ")" << std::endl;

    for(u32 i = 0 ; i < iboc_dim.x(); i++){
        for(u32 j = 0 ; j < iboc_dim.y(); j++){
            for(u32 k = 0 ; k < iboc_dim.z(); k++){

                vec3 r_a = {
                    2*i + ((j+k) % 2),
                    sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                    2*sycl::sqrt(6.)*k/3
                };

                r_a *= r_particle;
                r_a += box_min;

                if(selector(r_a)) part_pusher(r_a, r_particle);

            }
        }
    }

}