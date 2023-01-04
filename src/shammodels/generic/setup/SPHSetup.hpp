// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/patch/base/patchdata.hpp"
#include "shamrock/patch/base/patchdata_field.hpp"
#include "shamrock/patch/utility/serialpatchtree.hpp"
#include "shamrock/patch/scheduler/scheduler_mpi.hpp"
#include "shamsys/mpi_handler.hpp"
#include "shamrock/patch/comm/patch_object_mover.hpp"
#include "shamsys/sycl_mpi_interop.hpp"
#include <memory>
#include <vector>


//%Impl status : Good

template <class flt>
class SPHSetup{

    using vec3 = sycl::vec<flt, 3>;

    PatchScheduler & sched;

    bool periodic_mode;

    u64 part_cnt = 0;


    public:

    SPHSetup(PatchScheduler & scheduler,bool periodic) : sched(scheduler), periodic_mode(periodic) {}

    
    inline void init_setup(){
        if (mpi_handler::world_rank == 0) {
            Patch root;

            root.node_owner_id = mpi_handler::world_rank;

            root.x_min = 0;
            root.y_min = 0;
            root.z_min = 0;

            root.x_max = HilbertLB::max_box_sz;
            root.y_max = HilbertLB::max_box_sz;
            root.z_max = HilbertLB::max_box_sz;

            root.pack_node_index = u64_max;

            PatchData pdat(sched.pdl);

            root.data_count = pdat.get_obj_cnt();
            root.load_value = pdat.get_obj_cnt();

            sched.add_patch(root,pdat);  

        } else {
            sched.patch_list._next_patch_id++;
        }  

        mpi::barrier(MPI_COMM_WORLD);

        sched.owned_patch_id = sched.patch_list.build_local();

        sched.patch_list.build_global();

        sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);

        std::cout << "build local" << std::endl;
        sched.owned_patch_id = sched.patch_list.build_local();
        sched.patch_list.build_local_idx_map();
        sched.update_local_dtcnt_value();
        sched.update_local_load_value();
    }

    template<class LambdaSelect>
    inline void add_particules_fcc(flt dr, std::tuple<vec3,vec3> box,LambdaSelect && selector){

        if(mpi_handler::world_rank == 0){
            std::vector<vec3> vec;

            add_particles_fcc(
                dr, 
                box , 
                selector, 
                [&](f32_3 r,f32 h){
                    vec.push_back(r); 
                });

            std::cout << ">>> adding : " << vec.size() << " objects" << std::endl;

            PatchData tmp(sched.pdl);
            tmp.resize(vec.size());

            part_cnt+= vec.size();

            PatchDataField<vec3> & f = tmp.get_field<vec3>(sched.pdl.get_field_idx<vec3>("xyz"));

            sycl::buffer<vec3> buf (vec.data(),vec.size());

            f.override(buf);

            if(sched.owned_patch_id.empty()) throw shamrock_exc("the scheduler does not have patch in that rank");

            u64 insert_id = *sched.owned_patch_id.begin();

            sched.patch_data.owned_data.at(insert_id).insert_elements(tmp);
        }


        //TODO apply position modulo here

        sched.scheduler_step(false, false);

        {
            SerialPatchTree<vec3> sptree(sched.patch_tree, sched.get_box_tranform<vec3>());
            sptree.attach_buf();
            reatribute_particles(sched, sptree, periodic_mode);
        }

        sched.scheduler_step(true, true);

        for (auto & [pid,pdat] : sched.patch_data.owned_data) {

            PatchDataField<flt> & f = pdat.template get_field<flt>(sched.pdl.get_field_idx<flt>("hpart"));

            f.override(dr);
        }
        
    }


    inline flt get_part_mass(flt tot_mass){

        u64 part = 0;


        mpi::allreduce(&part_cnt, &part, 1, mpi_type_u64, MPI_SUM, MPI_COMM_WORLD);

        return tot_mass/part;
    }

};