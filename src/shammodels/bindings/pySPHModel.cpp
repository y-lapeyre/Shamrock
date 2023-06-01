// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambindings/pybindaliases.hpp"
#include "shammodels/VariantSPHModel.hpp"
#include <memory>

Register_pymod(pysphmodel){

    using namespace shammodels;

    py::class_<SPHModelVariantBindings>(m, "SPHModel")
        .def(
            py::init([](ShamrockCtx & ctx, std::string vector_type, std::string kernel) {
                return std::make_unique<SPHModelVariantBindings>(ctx, vector_type,kernel);
            }),
            py::kw_only(),
            py::arg("context"),
            py::arg("vector_type"),
            py::arg("sph_kernel")
        )
        .def("init_scheduler",&SPHModelVariantBindings::init_scheduler)
        .def("evolve",[](SPHModelVariantBindings & self, f64 dt,bool physics_on, bool do_dump, std::string dump_name, bool debug_dump){
            return self.evolve_once(dt,physics_on, 
                do_dump,
                dump_name,
                debug_dump
            );
        })
        .def("set_cfl_cour",&SPHModelVariantBindings::set_cfl_cour)   
        .def("set_cfl_force",&SPHModelVariantBindings::set_cfl_force)   
        .def("set_particle_mass",&SPHModelVariantBindings::set_particle_mass)   
        
        
        ;

}