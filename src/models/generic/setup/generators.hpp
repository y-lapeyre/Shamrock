#pragma once

#include "aliases.hpp"
namespace generic::setup::generators {

    template<class flt>
    inline sycl::vec<flt, 3> get_box_dim(flt r_particle, u32 xcnt, u32 ycnt, u32 zcnt){

        using vec3 = sycl::vec<flt, 3>;

        u32 im = xcnt;
        u32 jm = ycnt;
        u32 km = zcnt;


        auto get_pos = [&](u32 i, u32 j, u32 k) -> vec3{
            vec3 r_a = {
                2*i + ((j+k) % 2),
                sycl::sqrt(3.)*(j + (1./3.)*(k % 2)),
                2*sycl::sqrt(6.)*k/3
            };

            r_a *= r_particle;

            return r_a;
        };

        return get_pos(im,jm,km);
    }

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

        std::cout << "get_ideal_box_idim :" << i << " " << j << " " << k << std::endl;

        i -= i%2;
        j -= j%2;
        k -= k%2;

        vec3 m1 = get_box_dim(r_particle, i, j, k);

        return {box_min, box_min + m1};

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

        std::cout << "part box size : (" << iboc_dim.x() << ", " << iboc_dim.y() << ", " << iboc_dim.z() << ")" << std::endl;
        u32 ix = std::ceil(iboc_dim.x());
        u32 iy = std::ceil(iboc_dim.y());
        u32 iz = std::ceil(iboc_dim.z());
        std::cout << "part box size : (" << ix << ", " << iy << ", " << iz << ")" << std::endl;

        if((iy % 2) != 0 && (iz % 2) != 0){
            std::cout << "Warning : particle count is odd on axis y or z -> this may lead to periodicity issues";
        }

        for(u32 i = 0 ; i < ix; i++){
            for(u32 j = 0 ; j < iy; j++){
                for(u32 k = 0 ; k < iz; k++){

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


} // namespace generic::setup::generators