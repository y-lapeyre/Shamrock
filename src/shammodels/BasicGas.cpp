// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "BasicGas.hpp"
namespace shammodels::sph {


    void BasicGas::dump_vtk(std::string dump_name) {

        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = 0; // TODO get_rank_count() in scheduler
        scheduler().for_each_patch_data(
            [&](u64 id_patch, Patch cur_p, PatchData &pdat) { num_obj += pdat.get_obj_cnt(); });

        // TODO aggregate field ?
        sycl::buffer<vec> pos(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(
                pos, get_check_ref(pdat.get_field<vec>(0).get_buf()), ptr, pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writer.write_points(pos, num_obj);
    }

    void BasicGas::evolve(f64 dt){

        //forward euler step f dt/2

        //forward euler step positions dt

        //forward euler step f dt/2

        // swap der

        // periodic box

        // update h

        // compute pressure

        // compute force

        // corrector 

        // if delta too big jump to compute force
    }

}