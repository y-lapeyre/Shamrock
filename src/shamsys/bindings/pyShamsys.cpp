#include "pybindaliases.hpp"


#include "pyNodeInstance.hpp"


SHAMROCK_PY_MODULE(shamrock, m){

    py::module sys_module = m.def_submodule("sys", "system handling part of shamrock");

    shamsys::instance::register_pymodules(sys_module);
}
