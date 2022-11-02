// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include <memory>
#include <vector>

enum BCType {
    Free, Periodic, PeriodicShearing, Fixed, Ghost, FixedGradient, AntiPeriodic
};

template<class flt>
class BoundaryConditions {public:

    BCType type;

    using vec = sycl::vec<flt,3>;
    using vec_box = std::tuple<vec,vec>;

    std::unique_ptr<vec_box> box_bc;


    /*
     * describe the schearing cd in the following way
     * std::tuple<vec_box, u32_3, u32, vec> equivalent to [shearing vec, field id (flt type x 3), vec]
     * applied like this : field[fid] = field[fid] + vec*dot(pvec,svec);
     * svec is the shearing vector (argument)
     * pvec is the vector that discribe the periodicity
     */
    //std::vector<std::tuple<u32_3, u32, vec>> shear_cd;

    inline void set_mode(BCType bc){
        type = bc;

        if (bc == PeriodicShearing) {
            throw shamrock_exc("[BoundaryConditions] Shearing periodic mode not implemented");
        }

        if (bc == Fixed) {
            throw shamrock_exc("[BoundaryConditions] Dirichelt mode not implemented");
        }
    }

    inline void set_box(vec_box b){

        if(type == Free){ 
            throw shamrock_exc("[BoundaryConditions] Can not set box size with free boundary conditions");
        }

        box_bc = std::make_unique<vec_box>();
        *box_bc = b;
    }







};