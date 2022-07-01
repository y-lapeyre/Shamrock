#pragma once

#include "aliases.hpp"

template<class flt> 
inline std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> get_ideal_fcc_box(flt r_particle, 
    std::tuple<sycl::vec<flt, 3>,sycl::vec<flt, 3>> box){

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

    u32 i = iboc_dim.x();
    u32 j = iboc_dim.y();
    u32 k = iboc_dim.z();

    i -= i%6;
    j -= j%6;
    k -= k%2;


    auto get_pos = [&](u32 i, u32 j, u32 k){
        vec3 r_a = {
            2*i + ((j+k) % 2),
            sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
            2*sycl::sqrt(6.)*k/3
        };

        r_a *= r_particle;
        r_a += box_min;

        return r_a;
    };

    vec3 m1 = get_pos(i+1,j+1,k+1);

    return {box_min, m1};

}


template<class flt>
inline sycl::vec<flt, 3> get_box_dim(flt r_particle, u32 xcnt, u32 ycnt, u32 zcnt){

    using vec3 = sycl::vec<flt, 3>;

    u32 i = xcnt;
    u32 j = ycnt;
    u32 k = zcnt;

    auto get_pos = [&](u32 i, u32 j, u32 k){
        vec3 r_a = {
            2*i + ((j+k) % 2),
            sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
            2*sycl::sqrt(6.)*k/3
        };

        r_a *= r_particle;

        return r_a;
    };

    return get_pos(i+1,j+1,k+1);
}


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