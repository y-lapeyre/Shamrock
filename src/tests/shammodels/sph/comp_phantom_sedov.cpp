// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambase/fortran_io.hpp"
#include "shambase/memory.hpp"
#include "shambase/time.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/shamtest.hpp"
#include <builtins.hpp>
#include <vector>

shammodels::sph::PhantomDump load_dump(std::string file) {

    shambase::FortranIOFile fdat = shambase::load_fortran_file(file);

    return shammodels::sph::PhantomDump::from_file(fdat);
}

template<class T>
std::vector<T> fetch_data(std::string key, shamrock::patch::PatchData &pdat) {

    std::vector<T> vec;

    auto appender = [&](auto &field) {
        if (field.get_name() == key) {

            {
                sycl::host_accessor acc{shambase::get_check_ref(field.get_buf())};
                u32 len = field.size();

                for (u32 i = 0; i < len; i++) {
                    vec.push_back(acc[i]);
                }
            }
        }
    };

    pdat.for_each_field<T>([&](auto &field) {
        appender(field);
    });

    return vec;
}

void compare_results(
    std::string name,
    shamrock::patch::PatchData &pdat,
    shammodels::sph::PhantomDump &ref_file,
    f64 pmass) {

    std::vector<f64> sham_uint  = fetch_data<f64>("uint", pdat);
    std::vector<f64> sham_hpart = fetch_data<f64>("hpart", pdat);
    std::vector<f64> sham_alpha = fetch_data<f64>("alpha_AV", pdat);
    std::vector<f64> sham_x, sham_y, sham_z;
    std::vector<f64> sham_vx, sham_vy, sham_vz;
    std::vector<f64> sham_r, sham_vr;

    {
        std::vector<f64_3> xyz  = fetch_data<f64_3>("xyz", pdat);
        std::vector<f64_3> vxyz = fetch_data<f64_3>("vxyz", pdat);
        for (auto vec : xyz) {
            sham_x.push_back(vec.x());
            sham_y.push_back(vec.y());
            sham_z.push_back(vec.z());
            sham_r.push_back(sycl::length(vec));
        }

        for (auto vec : vxyz) {
            sham_vx.push_back(vec.x());
            sham_vy.push_back(vec.y());
            sham_vz.push_back(vec.z());
            sham_vr.push_back(sycl::length(vec));
        }
    }

    std::vector<f64> x, y, z, vx, vy, vz;
    std::vector<f64> h, u, alpha;

    ref_file.blocks[0].fill_vec("x", x);
    ref_file.blocks[0].fill_vec("y", y);
    ref_file.blocks[0].fill_vec("z", z);
    ref_file.blocks[0].fill_vec("h", h);

    ref_file.blocks[0].fill_vec("vx", vx);
    ref_file.blocks[0].fill_vec("vy", vy);
    ref_file.blocks[0].fill_vec("vz", vz);

    ref_file.blocks[0].fill_vec("u", u);
    ref_file.blocks[0].fill_vec("alpha", alpha);

    // f64 dxyz_max = 0;
    // for (u32 i = 0; i < 174000; i++) {
    //     dxyz_max = sham::max(dxyz_max, sham::abs(x[i] - sham_x[i]));
    //     dxyz_max = sham::max(dxyz_max, sham::abs(y[i] - sham_y[i]));
    //     dxyz_max = sham::max(dxyz_max, sham::abs(z[i] - sham_z[i]));
    // }
    // logger::raw_ln(dxyz_max);

    std::vector<f64> ph_r{};
    for (u32 i = 0; i < x.size(); i++) {
        ph_r.push_back(sycl::length(f64_3{x[i], y[i], z[i]}));
    }

    std::vector<f64> ph_vr{};
    for (u32 i = 0; i < x.size(); i++) {
        ph_vr.push_back(sycl::length(f64_3{vx[i], vy[i], vz[i]}));
    }

    PyScriptHandle hdnl{};

    hdnl.data()["pmass"] = pmass;

    hdnl.data()["ph_r"]   = ph_r;
    hdnl.data()["sham_r"] = sham_r;

    hdnl.data()["ph_h"]   = h;
    hdnl.data()["sham_h"] = sham_hpart;

    hdnl.data()["ph_u"]   = u;
    hdnl.data()["sham_u"] = sham_uint;

    hdnl.data()["ph_vr"]   = ph_vr;
    hdnl.data()["sham_vr"] = sham_vr;

    hdnl.data()["ph_alpha"]   = alpha;
    hdnl.data()["sham_alpha"] = sham_alpha;
    hdnl.data()["name"]       = name;

    hdnl.exec(R"(
        import numpy as np
        import matplotlib.pyplot as plt

        plt.style.use('custom_style.mplstyle')

        fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=125)

        ph_rho = pmass*(1.2/np.array(ph_h))**3
        sham_rho = pmass*(1.2/np.array(sham_h))**3

        smarker = 1
        print(len(ph_r), len(ph_rho))
        axs[0,0].scatter(ph_r,ph_rho,s=smarker,c = 'red', rasterized=True,label = "phantom")
        axs[0,1].scatter(ph_r,ph_u,s=smarker,c = 'red', rasterized=True)
        axs[1,0].scatter(ph_r,ph_vr,s=smarker,c = 'red', rasterized=True)
        axs[1,1].scatter(ph_r,ph_alpha, s=smarker,c = 'red', rasterized=True)

        axs[0,0].scatter(sham_r,sham_rho,s=smarker,c = 'black', rasterized=True,label = "shamrock")
        axs[0,1].scatter(sham_r,sham_u,s=smarker,c = 'black', rasterized=True)
        axs[1,0].scatter(sham_r,sham_vr,s=smarker,c = 'black', rasterized=True)
        axs[1,1].scatter(sham_r,sham_alpha, s=smarker,c = 'black', rasterized=True)

        
        axs[0,0].set_ylabel(r"$\rho$")
        axs[1,0].set_ylabel(r"$vr$")
        axs[0,1].set_ylabel(r"$u$")
        axs[1,1].set_ylabel(r"$\alpha$")

        axs[0,0].set_xlabel("$r$")
        axs[1,0].set_xlabel("$r$")
        axs[0,1].set_xlabel("$r$")
        axs[1,1].set_xlabel("$r$")

        axs[0,0].set_xlim(0,0.4)
        axs[1,0].set_xlim(0,0.4)
        axs[0,1].set_xlim(0,0.4)
        axs[1,1].set_xlim(0,0.4)

        axs[0,0].legend()

        plt.tight_layout()  

        plt.savefig("tests/figures/"+name, dpi = 300)

    )");

    // TEX_REPORT(R"==(
    //
    //    \begin{figure}[ht!]
    //    \center
    //    \includegraphics[width=0.95\linewidth]{figures/sph_kernels.pdf}
    //    \caption{SPH kernels implemented in shamrock}
    //    \end{figure}
    //
    //)==")
}

// 16 cores phantom (i7-10700) = 5min 27sec (1000 iter)
TestStart(
    ValidationTest, "shammodels/sph/sedov_blast_phantom_fix_dt", comp_sedov_phantom_fix_dt, 1) {

    f64 start_t = 0;
    f64 dt      = 1e-5;

    f64 end_t = 0.001;

    std::string ref_file_start = "reference-files/sedov_blast_phantom/blast_00000";
    std::string ref_file_0001  = "reference-files/sedov_blast_phantom/blast_00001";
    std::string ref_file_0010  = "reference-files/sedov_blast_phantom/blast_00010";
    std::string ref_file_0100  = "reference-files/sedov_blast_phantom/blast_00100";
    std::string ref_file_1000  = "reference-files/sedov_blast_phantom/blast_01000";

    using namespace shammodels::sph;

    PhantomDump dump_start = load_dump(ref_file_start);
    PhantomDump dump_0001  = load_dump(ref_file_0001);
    PhantomDump dump_0010  = load_dump(ref_file_0010);
    PhantomDump dump_0100  = load_dump(ref_file_0100);
    PhantomDump dump_1000  = load_dump(ref_file_1000);

    ShamrockCtx ctx{};
    ctx.pdata_layout_new();

    Model<f64_3, shammath::M4> model{ctx};
    auto cfg                   = model.gen_config_from_phantom_dump(dump_start, false);
    model.solver.solver_config = cfg;
    model.solver.solver_config.print_status();

    model.init_scheduler(1e7, 1);

    model.init_from_phantom_dump(dump_start);

    f64 t = start_t;
    model.evolve_once(t, 0, false, "", false);
    for (; t < dt * 1; t += dt) {
        model.evolve_once(t, dt, false, "", false);
    }
    {
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered_result =
            ctx.allgather_data();
        shamrock::patch::PatchData &pdat_end = shambase::get_check_ref(gathered_result[0]);
        compare_results(
            "shamrock_phantom_sedov_fix_dt_1step.pdf",
            pdat_end,
            dump_0001,
            model.solver.solver_config.gpart_mass);
    }

    for (; t < dt * 10; t += dt) {
        model.evolve_once(t, dt, false, "", false);
    }
    {
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered_result =
            ctx.allgather_data();
        shamrock::patch::PatchData &pdat_end = shambase::get_check_ref(gathered_result[0]);
        compare_results(
            "shamrock_phantom_sedov_fix_dt_10step.pdf",
            pdat_end,
            dump_0010,
            model.solver.solver_config.gpart_mass);
    }

    for (; t < dt * 100; t += dt) {
        model.evolve_once(t, dt, false, "", false);
    }

    {
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered_result =
            ctx.allgather_data();
        shamrock::patch::PatchData &pdat_end = shambase::get_check_ref(gathered_result[0]);
        compare_results(
            "shamrock_phantom_sedov_fix_dt_100step.pdf",
            pdat_end,
            dump_0100,
            model.solver.solver_config.gpart_mass);
    }
    for (; t < dt * 1000; t += dt) {
        model.evolve_once(t, dt, false, "", false);
    }
    {
        std::vector<std::unique_ptr<shamrock::patch::PatchData>> gathered_result =
            ctx.allgather_data();
        shamrock::patch::PatchData &pdat_end = shambase::get_check_ref(gathered_result[0]);
        compare_results(
            "shamrock_phantom_sedov_fix_dt_1000step.pdf",
            pdat_end,
            dump_1000,
            model.solver.solver_config.gpart_mass);
    }
}