// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shamphys/SodTube.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <memory>
#include <random>

template<class Tvec, template<class> class SPHKernel>
void add_instance(py::module &m, std::string name_config, std::string name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T = Model<Tvec, SPHKernel>;

    using TAnalysisSodTube = shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>;
    using TSPHSetup        = shammodels::sph::modules::SPHSetup<Tvec, SPHKernel>;
    using TConfig          = typename T::Solver::Config;

    logger::debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
    logger::debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

    py::class_<TConfig>(m, name_config.c_str())
        .def("print_status", &TConfig::print_status)
        .def("set_tree_reduction_level", &TConfig::set_tree_reduction_level)
        .def("set_two_stage_search", &TConfig::set_two_stage_search)
        .def("set_max_neigh_cache_size", &TConfig::set_max_neigh_cache_size)
        .def("set_eos_isothermal", &TConfig::set_eos_isothermal)
        .def("set_eos_adiabatic", &TConfig::set_eos_adiabatic)
        .def("set_eos_locally_isothermal", &TConfig::set_eos_locally_isothermal)
        .def(
            "set_eos_locally_isothermalLP07",
            [](TConfig &self, Tscal cs0, Tscal q, Tscal r0) {
                self.set_eos_locally_isothermalLP07(cs0, q, r0);
            },
            py::kw_only(),
            py::arg("cs0"),
            py::arg("q"),
            py::arg("r0"))
        .def(
            "set_eos_locally_isothermalFA2014",
            [](TConfig &self, Tscal h_over_r) {
                self.set_eos_locally_isothermalFA2014(h_over_r);
            },
            py::kw_only(),
            py::arg("h_over_r"))
        .def("set_artif_viscosity_None", &TConfig::set_artif_viscosity_None)
        .def(
            "to_json",
            [](TConfig &self) {
                return nlohmann::json{self}.dump(4);
            })
        .def(
            "set_artif_viscosity_Constant",
            [](TConfig &self, Tscal alpha_u, Tscal alpha_AV, Tscal beta_AV) {
                self.set_artif_viscosity_Constant({alpha_u, alpha_AV, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_u"),
            py::arg("alpha_AV"),
            py::arg("beta_AV"))
        .def(
            "set_artif_viscosity_VaryingMM97",
            [](TConfig &self,
               Tscal alpha_min,
               Tscal alpha_max,
               Tscal sigma_decay,
               Tscal alpha_u,
               Tscal beta_AV) {
                self.set_artif_viscosity_VaryingMM97(
                    {alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_min"),
            py::arg("alpha_max"),
            py::arg("sigma_decay"),
            py::arg("alpha_u"),
            py::arg("beta_AV"))
        .def(
            "set_artif_viscosity_VaryingCD10",
            [](TConfig &self,
               Tscal alpha_min,
               Tscal alpha_max,
               Tscal sigma_decay,
               Tscal alpha_u,
               Tscal beta_AV) {
                self.set_artif_viscosity_VaryingCD10(
                    {alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_min"),
            py::arg("alpha_max"),
            py::arg("sigma_decay"),
            py::arg("alpha_u"),
            py::arg("beta_AV"))
        .def(
            "set_artif_viscosity_ConstantDisc",
            [](TConfig &self, Tscal alpha_AV, Tscal alpha_u, Tscal beta_AV) {
                self.set_artif_viscosity_ConstantDisc({alpha_AV, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_AV"),
            py::arg("alpha_u"),
            py::arg("beta_AV"))
        .def(
            "set_IdealMHD",
            [](TConfig &self, Tscal sigma_mhd, Tscal sigma_u) {
                self.set_IdealMHD({sigma_mhd, sigma_u});
            },
            py::kw_only(),
            py::arg("sigma_mhd"),
            py::arg("sigma_u"))
        .def("set_boundary_free", &TConfig::set_boundary_free)
        .def("set_boundary_periodic", &TConfig::set_boundary_periodic)
        .def("set_boundary_shearing_periodic", &TConfig::set_boundary_shearing_periodic)

        .def("add_ext_force_point_mass", &TConfig::add_ext_force_point_mass)
        .def(
            "add_ext_force_lense_thirring",
            [](TConfig &self, Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
                self.add_ext_force_lense_thirring(central_mass, Racc, a_spin, dir_spin);
            },
            py::kw_only(),
            py::arg("central_mass"),
            py::arg("Racc"),
            py::arg("a_spin"),
            py::arg("dir_spin"))
        .def(
            "add_ext_force_shearing_box",
            [](TConfig &self, Tscal Omega_0, Tscal eta, Tscal q) {
                self.add_ext_force_shearing_box(Omega_0, eta, q);
            },
            py::kw_only(),
            py::arg("Omega_0"),
            py::arg("eta"),
            py::arg("q"))
        .def("set_units", &TConfig::set_units)
        .def("set_cfl_multipler", &TConfig::set_cfl_multipler)
        .def("set_cfl_mult_stiffness", &TConfig::set_cfl_mult_stiffness);

    std::string sod_tube_analysis_name = name_model + "_AnalysisSodTube";
    py::class_<TAnalysisSodTube>(m, sod_tube_analysis_name.c_str())
        .def("compute_L2_dist", [](TAnalysisSodTube &self) -> std::tuple<Tscal, Tvec, Tscal> {
            auto ret = self.compute_L2_dist();
            return {ret.rho, ret.v, ret.P};
        });

    std::string setup_name = name_model + "_SPHSetup";
    py::class_<TSPHSetup>(m, setup_name.c_str())
        .def(
            "make_generator_lattice_hcp",
            [](TSPHSetup &self, Tscal dr, Tvec box_min, Tvec box_max) {
                return self.make_generator_lattice_hcp(dr, {box_min, box_max});
            })
        .def(
            "make_generator_disc_mc",
            [](TSPHSetup &self,
               Tscal part_mass,
               Tscal disc_mass,
               Tscal r_in,
               Tscal r_out,
               std::function<Tscal(Tscal)> sigma_profile,
               std::function<Tscal(Tscal)> H_profile,
               std::function<Tscal(Tscal)> rot_profile,
               std::function<Tscal(Tscal)> cs_profile,
               u64 random_seed) {
                return self.make_generator_disc_mc(
                    part_mass,
                    disc_mass,
                    r_in,
                    r_out,
                    sigma_profile,
                    H_profile,
                    rot_profile,
                    cs_profile,
                    std::mt19937(random_seed));
            },
            py::kw_only(),
            py::arg("part_mass"),
            py::arg("disc_mass"),
            py::arg("r_in"),
            py::arg("r_out"),
            py::arg("sigma_profile"),
            py::arg("H_profile"),
            py::arg("rot_profile"),
            py::arg("cs_profile"),
            py::arg("random_seed"))
        .def(
            "make_combiner_add",
            [](TSPHSetup &self,
               shammodels::sph::modules::SetupNodePtr parent1,
               shammodels::sph::modules::SetupNodePtr parent2) {
                return self.make_combiner_add(parent1, parent2);
            })
        .def(
            "apply_setup",
            [](TSPHSetup &self,
               shammodels::sph::modules::SetupNodePtr setup,
               bool part_reordering,
               std::optional<u32> insert_step) {
                return self.apply_setup(setup, part_reordering, insert_step);
            },
            py::arg("setup"),
            py::kw_only(),
            py::arg("part_reordering") = true,
            py::arg("insert_step")     = std::nullopt);

    py::class_<T>(m, name_model.c_str())
        .def(py::init([](ShamrockCtx &ctx) {
            return std::make_unique<T>(ctx);
        }))
        .def("init_scheduler", &T::init_scheduler)

        .def("evolve_once_override_time", &T::evolve_once_time_expl)
        .def("evolve_once", &T::evolve_once)
        .def(
            "evolve_until",
            [](T &self, f64 target_time, i32 niter_max) {
                return self.evolve_until(target_time, niter_max);
            },
            py::arg("target_time"),
            py::kw_only(),
            py::arg("niter_max") = -1)
        .def("timestep", &T::timestep)
        .def("set_cfl_cour", &T::set_cfl_cour)
        .def("set_cfl_force", &T::set_cfl_force)
        .def("set_particle_mass", &T::set_particle_mass)
        .def("rho_h", &T::rho_h)
        .def("get_hfact", &T::get_hfact)
        .def(
            "get_box_dim_fcc_3d",
            [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
                return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
            })
        .def(
            "get_ideal_fcc_box",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.get_ideal_fcc_box(dr, {box_min, box_max});
            })
        .def(
            "get_ideal_hcp_box",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.get_ideal_hcp_box(dr, {box_min, box_max});
            })
        .def(
            "resize_simulation_box",
            [](T &self, f64_3 box_min, f64_3 box_max) {
                return self.resize_simulation_box({box_min, box_max});
            })
        .def(
            "push_particle",
            [](T &self, std::vector<f64_3> pos, std::vector<f64> hpart, std::vector<f64> upart) {
                return self.push_particle(pos, hpart, upart);
            })
        .def(
            "push_particle_mhd",
            [](T &self,
               std::vector<f64_3> pos,
               std::vector<f64> hpart,
               std::vector<f64> upart,
               std::vector<f64_3> vel,
               std::vector<f64_3> BOR,
               std::vector<f64> POC) {
                return self.push_particle_mhd(pos, hpart, upart, vel, BOR, POC);
            })
        .def(
            "add_cube_fcc_3d",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.add_cube_fcc_3d(dr, {box_min, box_max});
            })
        .def(
            "add_cube_hcp_3d",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.add_cube_hcp_3d(dr, {box_min, box_max});
            })
        .def(
            "add_cube_hcp_3d_v2",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.add_cube_hcp_3d_v2(dr, {box_min, box_max});
            })
        .def(
            "add_disc_3d_keplerian",
            [](T &self,
               Tvec center,
               u32 Npart,
               Tscal p,
               Tscal rho_0,
               Tscal m,
               Tscal r_in,
               Tscal r_out,
               Tscal q,
               Tscal cmass) {
                return self.add_cube_disc_3d(center, Npart, p, rho_0, m, r_in, r_out, q, cmass);
            })
        .def(
            "add_disc_3d",
            [](T &self,
               Tvec center,
               Tscal central_mass,
               u32 Npart,
               Tscal r_in,
               Tscal r_out,
               Tscal disc_mass,
               Tscal p,
               Tscal H_r_in,
               Tscal q) {
                return self.add_disc_3d(
                    center, central_mass, Npart, r_in, r_out, disc_mass, p, H_r_in, q);
            })
        .def(
            "add_big_disc_3d",
            [](T &self,
               Tvec center,
               Tscal central_mass,
               u32 Npart,
               Tscal r_in,
               Tscal r_out,
               Tscal disc_mass,
               Tscal p,
               Tscal H_r_in,
               Tscal q,
               u16 seed) {
                self.add_big_disc_3d(
                    center,
                    central_mass,
                    Npart,
                    r_in,
                    r_out,
                    disc_mass,
                    p,
                    H_r_in,
                    q,
                    std::mt19937{seed});
                return disc_mass / Npart;
            })
        .def("get_total_part_count", &T::get_total_part_count)
        .def("total_mass_to_part_mass", &T::total_mass_to_part_mass)
        .def(
            "set_value_in_a_box",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 box_min,
               f64_3 box_max) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max});
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max});
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def(
            "set_value_in_sphere",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 center,
               f64 radius) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_value_in_sphere(field_name, val, center, radius);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_value_in_sphere(field_name, val, center, radius);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def("set_field_value_lambda_f64_3", &T::template set_field_value_lambda<f64_3>)
        .def("set_field_value_lambda_f64", &T::template set_field_value_lambda<f64>)
        .def("remap_positions", &T::remap_positions)
        //.def("set_field_value_lambda_f64_3",[](T&self,std::string field_name, const
        // std::function<f64_3 (Tscal, Tscal , Tscal)> pos_to_val){
        //    self.template set_field_value_lambda<f64_3>(field_name, [=](Tvec v){
        //        return pos_to_val(v.x(), v.y(),v.z());
        //    });
        //})
        .def(
            "add_kernel_value",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 center,
               f64 h_ker) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.add_kernel_value(field_name, val, center, h_ker);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.add_kernel_value(field_name, val, center, h_ker);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def(
            "get_sum",
            [](T &self, std::string field_name, std::string field_type) {
                if (field_type == "f64") {
                    return py::cast(self.template get_sum<f64>(field_name));
                } else if (field_type == "f64_3") {
                    return py::cast(self.template get_sum<f64_3>(field_name));
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def(
            "get_closest_part_to",
            [](T &self, f64_3 pos) -> f64_3 {
                return self.get_closest_part_to(pos);
            })
        .def(
            "gen_default_config",
            [](T &self) {
                return typename T::Solver::Config{};
            })
        .def(
            "get_current_config",
            [](T &self) {
                return self.solver.solver_config;
            })
        .def("set_solver_config", &T::set_solver_config)
        .def("add_sink", &T::add_sink)
        .def(
            "get_sinks",
            [](T &self) {
                py::list list_out;

                if (!self.solver.storage.sinks.is_empty()) {
                    for (auto &sink : self.solver.storage.sinks.get()) {
                        py::dict sink_dic;
                        sink_dic["pos"]              = sink.pos;
                        sink_dic["velocity"]         = sink.velocity;
                        sink_dic["sph_acceleration"] = sink.sph_acceleration;
                        sink_dic["ext_acceleration"] = sink.ext_acceleration;
                        sink_dic["mass"]             = sink.mass;
                        sink_dic["angular_momentum"] = sink.angular_momentum;
                        sink_dic["accretion_radius"] = sink.accretion_radius;
                        list_out.append(sink_dic);
                    }
                }

                return list_out;
            })
        .def(
            "render_cartesian_slice",
            [](T &self,
               std::string name,
               std::string field_type,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> std::variant<py::array_t<Tscal>> {
                if (field_type == "f64") {
                    py::array_t<Tscal> ret({ny, nx});

                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64> slice
                        = render.compute_slice(name, center, delta_x, delta_y, nx, ny)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                        }
                    }

                    return ret;
                }

                if (field_type == "f64_3") {
                    py::array_t<Tscal> ret({ny, nx, 3_u32});

                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64_3> slice
                        = render.compute_slice(name, center, delta_x, delta_y, nx, ny)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                            ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                            ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                        }
                    }

                    return ret;
                }

                shambase::throw_with_loc<std::runtime_error>("unknown slice type");
                return py::array_t<Tscal>({nx, ny});
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"))
        .def(
            "render_cartesian_column_integ",
            [](T &self,
               std::string name,
               std::string field_type,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> std::variant<py::array_t<Tscal>> {
                if (field_type == "f64") {
                    py::array_t<Tscal> ret({ny, nx});

                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64> slice
                        = render.compute_column_integ(name, center, delta_x, delta_y, nx, ny)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                        }
                    }

                    return ret;
                }

                if (field_type == "f64_3") {
                    py::array_t<Tscal> ret({ny, nx, 3_u32});

                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64_3> slice
                        = render.compute_column_integ(name, center, delta_x, delta_y, nx, ny)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                            ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                            ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                        }
                    }

                    return ret;
                }

                shambase::throw_with_loc<std::runtime_error>("unknown slice type");
                return py::array_t<Tscal>({nx, ny});
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"))
        .def(
            "gen_config_from_phantom_dump",
            [](T &self, PhantomDump &dump, bool bypass_error) {
                return self.gen_config_from_phantom_dump(dump, bypass_error);
            },
            py::arg("dump"),
            py::arg("bypass_error") = false,
            R"==(
    This function generate a shamrock sph solver config from a phantom dump

    Parameters
    ----------
    PhantomDump dump
    bypass_error = false (default) bypass any error in the config
)==")
        .def(
            "init_from_phantom_dump",
            [](T &self, PhantomDump &dump) {
                self.init_from_phantom_dump(dump);
            })
        .def(
            "make_phantom_dump",
            [](T &self) {
                return self.make_phantom_dump();
            })
        .def("do_vtk_dump", &T::do_vtk_dump)
        .def("set_debug_dump", &T::set_debug_dump)
        .def("solver_logs_last_rate", &T::solver_logs_last_rate)
        .def("solver_logs_last_obj_count", &T::solver_logs_last_obj_count)
        .def(
            "get_time",
            [](T &self) {
                return self.solver.solver_config.get_time();
            })
        .def(
            "get_dt",
            [](T &self) {
                return self.solver.solver_config.get_dt_sph();
            })
        .def(
            "set_time",
            [](T &self, Tscal t) {
                return self.solver.solver_config.set_time(t);
            })
        .def(
            "set_next_dt",
            [](T &self, Tscal dt) {
                return self.solver.solver_config.set_next_dt(dt);
            })
        .def(
            "set_cfl_multipler",
            [](T &self, Tscal lambda) {
                return self.solver.solver_config.set_cfl_multipler(lambda);
            })
        .def(
            "set_cfl_mult_stiffness",
            [](T &self, Tscal cstiff) {
                return self.solver.solver_config.set_cfl_mult_stiffness(cstiff);
            })
        .def("change_htolerance", &T::change_htolerance)
        .def(
            "make_analysis_sodtube",
            [](T &self,
               shamphys::SodTube sod,
               Tvec direction,
               Tscal time_val,
               Tscal x_ref,
               Tscal x_min,
               Tscal x_max) {
                return std::make_unique<TAnalysisSodTube>(
                    self.ctx,
                    self.solver.solver_config,
                    self.solver.storage,
                    sod,
                    direction,
                    time_val,
                    x_ref,
                    x_min,
                    x_max);
            })
        .def("load_from_dump", &T::load_from_dump)
        .def("dump", &T::dump)
        .def("get_setup", &T::get_setup);
    ;
}

Register_pymod(pysphmodel) {

    using namespace shammodels::sph;

    add_instance<f64_3, shammath::M4>(m, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    add_instance<f64_3, shammath::M6>(m, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");
    add_instance<f64_3, shammath::M8>(m, "SPHModel_f64_3_M8_SolverConfig", "SPHModel_f64_3_M8");

    using VariantSPHModelBind = std::variant<
        std::unique_ptr<Model<f64_3, shammath::M4>>,
        std::unique_ptr<Model<f64_3, shammath::M6>>,
        std::unique_ptr<Model<f64_3, shammath::M8>>>;

    m.def(
        "get_SPHModel",
        [](ShamrockCtx &ctx, std::string vector_type, std::string kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M6") {
                ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M8") {
                ret = std::make_unique<Model<f64_3, shammath::M8>>(ctx);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));

    py::class_<
        shammodels::sph::modules::ISPHSetupNode,
        std::shared_ptr<shammodels::sph::modules::ISPHSetupNode>>(m, "ISPHSetupNode")
        .def("get_dot", [](std::shared_ptr<shammodels::sph::modules::ISPHSetupNode> &self) {
            return self->get_dot();
        });

    py::class_<shammodels::sph::TimestepLog>(m, "TimestepLog")
        .def(py::init<>())
        .def_readwrite("rank", &shammodels::sph::TimestepLog::rank)
        .def_readwrite("rate", &shammodels::sph::TimestepLog::rate)
        .def_readwrite("npart", &shammodels::sph::TimestepLog::npart)
        .def_readwrite("tcompute", &shammodels::sph::TimestepLog::tcompute)
        .def("rate_sum", &shammodels::sph::TimestepLog::rate_sum)
        .def("npart_sum", &shammodels::sph::TimestepLog::npart_sum);
}
