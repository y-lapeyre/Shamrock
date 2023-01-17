#include "pybindaliases.hpp"


#include "pyNodeInstance.hpp"
#include "shamsys/legacy/log.hpp"


SHAMROCK_PY_MODULE(shamrock, m){

    py::module sys_module = m.def_submodule("sys", "system handling part of shamrock");

    m.def("change_loglevel",[](u32 loglevel){

        if (loglevel > 127) {
            throw std::invalid_argument("loglevel must be below 128");
        }

        if(loglevel == i8_max){
            logger::raw_ln("If you've seen spam in your life i can garantee you, this is worst");
        }

        logger::raw_ln("-> modified loglevel to",logger::loglevel,"enabled log types : ");

        logger::loglevel = loglevel;
        logger::print_active_level();

    }, R"pbdoc(

        Change the loglevel

    )pbdoc");

    shamsys::instance::register_pymodules(sys_module);
}
