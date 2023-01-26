// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/cmdopt.hpp"

#include "shambindings/pybindaliases.hpp"



namespace shamsys::instance {
    void register_pymodules(py::module & m){

        using namespace shamsys::instance;

        m.def("init",[](u32 alt_id, u32 compute_id){

            
            init(SyclInitInfo{alt_id,compute_id}, MPIInitInfo{opts::get_argc(),opts::get_argv()});

        }, R"pbdoc(

            The init function for shamrock node instance

        )pbdoc");


        m.def("close",[](){
            close();
        }, R"pbdoc(

            The close function for shamrock node instance

        )pbdoc");

        m.def("get_process_name",&get_process_name, R"pbdoc(

            Get the name of the process

        )pbdoc");

        m.def("world_rank",[](){
            return world_rank;
        }, R"pbdoc(
            Get the world rank
        )pbdoc");

        m.def("world_size",[](){
            return world_size;
        }, R"pbdoc(
            Get the world size
        )pbdoc");

        m.def("is_initialized",[](){
            return is_initialized();
        }, R"pbdoc(
            Return true if the node instance is initialized
        )pbdoc");
    }
}






