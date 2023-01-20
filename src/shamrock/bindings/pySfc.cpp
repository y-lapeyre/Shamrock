// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambindings/pybindaliases.hpp"

#include <pybind11/stl.h>

#include "shamrock/sfc/bmi.hpp"
#include "shamrock/sfc/morton.hpp"
#include "shamrock/sfc/hilbert.hpp"
#include <variant>

using namespace shamrock::sfc;

enum ReprType {
    _u32,_u64
};

Register_pymod(sfclibinit){

    py::module sfc_module = m.def_submodule("sfc", "Space Filling curve Library");

    py::enum_<ReprType>(sfc_module, "bitrepr")
        .value("u32", _u32)
        .value("u64", _u64)
        .export_values();


    sfc_module.def("coord_to_morton_3d_u32",[](f64 x, f64 y, f64 z){
        return MortonCodes<u32, 3>::coord_to_morton(x, y, z);
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");

    sfc_module.def("coord_to_morton_3d_u64",[](f64 x, f64 y, f64 z){
        return MortonCodes<u64, 3>::coord_to_morton(x, y, z);
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");


    sfc_module.def("morton_to_icoord_3d_u64",[](u64 m) -> std::array<u32,3>{
        auto ret = MortonCodes<u64, 3>::morton_to_icoord(m);
        return {u32{ret.x()},u32{ret.y()},u32{ret.z()}};
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");

    sfc_module.def("morton_to_coord_3d_u64",[](u64 m) -> std::array<f64,3>{
        auto ret = MortonCodes<u64, 3>::morton_to_icoord(m);
        f64 mult = MortonCodes<u64, 3>::max_val+1;
        return {f64(ret.x())/mult,f64(ret.y())/mult,f64(ret.z())/mult};
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");


    sfc_module.def("morton_to_icoord_3d_u32",[](u32 m) -> std::array<u32,3>{
        auto ret = MortonCodes<u32, 3>::morton_to_icoord(m);
        return {u32{ret.x()},u32{ret.y()},u32{ret.z()}};
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");

    sfc_module.def("morton_to_coord_3d_u32",[](u32 m) -> std::array<f64,3>{
        auto ret = MortonCodes<u32, 3>::morton_to_icoord(m);
        f64 mult = MortonCodes<u32, 3>::max_val+1;
        return {f64(ret.x())/mult,f64(ret.y())/mult,f64(ret.z())/mult};
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to morton codes
    )pbdoc");





    sfc_module.def("coord_to_hilbert_3d_u64",[](f64 x, f64 y, f64 z){
        return HilbertCurve<u64, 3>::coord_to_hilbert(x, y, z);
    }, R"pbdoc(
        Convert a 3d coordinate in the unit cube to hilbert codes
    )pbdoc");

}
