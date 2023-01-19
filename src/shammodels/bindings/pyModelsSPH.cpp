#include "shambindings/pybindaliases.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>

#include "NamedBasicSPHGasUint.hpp"


Register_pymod(pynamedsphmodels){

    py::class_<NamedBasicSPHUinterne>(m, "BasicSPHUinterne")
        .def(
            py::init([](std::string kernel, std::string precision) {
                return std::make_unique<NamedBasicSPHUinterne>(kernel, precision);
            }),
            py::kw_only(),
            py::arg("kernel"),
            py::arg("precision")
        )
        .def("init",&NamedBasicSPHUinterne::init)   
        .def("evolve",&NamedBasicSPHUinterne::evolve)   
        .def("simulate_until",&NamedBasicSPHUinterne::simulate_until)   
        .def("close",&NamedBasicSPHUinterne::close)   
        .def("set_cfl_cour",&NamedBasicSPHUinterne::set_cfl_cour)   
        .def("set_cfl_force",&NamedBasicSPHUinterne::set_cfl_force)   
        .def("set_particle_mass",&NamedBasicSPHUinterne::set_particle_mass)   
    ;


}