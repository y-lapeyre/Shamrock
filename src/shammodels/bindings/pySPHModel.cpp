// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "shammodels/VariantSPHModel.hpp"
#include <memory>

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"




Register_pymod(pysphmodel) {

    using namespace shammodels;

    using T = SPHModel<f64_3, shamrock::sph::kernels::M4>;

    py::class_<T>(m, "SPHModel_f64_3_M4")
        .def(py::init([](ShamrockCtx &ctx) { return std::make_unique<T>(ctx); }))
        .def("init_scheduler", &T::init_scheduler)
        .def("evolve", &T::evolve_once)
        .def("set_cfl_cour", &T::set_cfl_cour)
        .def("set_cfl_force", &T::set_cfl_force)
        .def("set_particle_mass", &T::set_particle_mass)
        .def("get_box_dim_fcc_3d", [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
            return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
        })
        .def("get_ideal_fcc_box", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.get_ideal_fcc_box(dr, {box_min,box_max});
        })
        ;





    using VariantSPHModelBind =
        std::variant<std::unique_ptr<SPHModel<f64_3, shamrock::sph::kernels::M4>>>;

    m.def(
        "get_SPHModel",
        [](ShamrockCtx &ctx, std::string vector_type, std::string kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<SPHModel<f64_3, shamrock::sph::kernels::M4>>(ctx);
            } else {
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));
}