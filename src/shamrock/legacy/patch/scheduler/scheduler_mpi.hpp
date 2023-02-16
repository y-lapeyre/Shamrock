// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file scheduler_mpi.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief MPI scheduler
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <fstream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"
//#include "shamrock/legacy/patch/patchdata_buffer.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/legacy/patch/base/patchtree.hpp"
#include "scheduler_patch_list.hpp"
#include "scheduler_patch_data.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

#include "shamrock/math/integerManip.hpp"

/**
 * @brief The MPI scheduler
 * 
 */
class PatchScheduler{
    
    using LoadBalancer = shamrock::scheduler::HilbertLoadBalance<u64>;
    
    public:

    static constexpr u64 max_axis_patch_coord = LoadBalancer::max_box_sz;
    static constexpr u64 max_axis_patch_coord_lenght = LoadBalancer::max_box_sz+1;

    using PatchTree = shamrock::scheduler::PatchTree;

    shamrock::patch::PatchDataLayout & pdl;

    u64 crit_patch_split; ///< splitting limit (if load value > crit_patch_split => patch split)
    u64 crit_patch_merge; ///< merging limit (if load value < crit_patch_merge => patch merge)


    SchedulerPatchList patch_list; ///< handle the list of the patches of the scheduler
    SchedulerPatchData patch_data; ///< handle the data of the patches of the scheduler
    PatchTree patch_tree; ///< handle the tree structure of the patches

    //using unordered set is not an issue since we use the find command after 
    std::unordered_set<u64>  owned_patch_id; ///< list of owned patch ids updated with (owned_patch_id = patch_list.build_local())


    


    /**
     * @brief scheduler step
     * 
     * @param do_split_merge 
     * @param do_load_balancing 
     */
    void scheduler_step(bool do_split_merge,bool do_load_balancing);
    
    
    void init_mpi_required_types();
    
    void free_mpi_required_types();

    PatchScheduler(shamrock::patch::PatchDataLayout & pdl, u64 crit_split,u64 crit_merge);

    ~PatchScheduler();



    std::string dump_status();


    inline void update_local_dtcnt_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].data_count = patch_data.owned_data.at(id).get_obj_cnt();
        }
    }

    inline void update_local_load_value(){
        for(u64 id : owned_patch_id){
            patch_list.local[patch_list.id_patch_to_local_idx[id]].load_value = patch_data.owned_data.at(id).get_obj_cnt();
        }
    }


    template<class vectype>
    std::tuple<vectype,vectype> get_box_tranform();

    template<class vectype>
    std::tuple<vectype,vectype> get_box_volume();

    bool should_resize_box(bool node_in);


    /**
     * @brief modify the bounding box of the patch domain
     * 
     * @tparam vectype 
     * @param bmin 
     * @param bmax 
     */
    template<class vectype>
    void set_coord_domain_bound(vectype bmin, vectype bmax){

        if(!pdl.check_main_field_type<vectype>()){
            std::invalid_argument(
                std::string("the main field is not of the correct type to call this function\n")+
                "fct called : " + __PRETTY_FUNCTION__ +
                "current patch data layout : "+
                pdl.get_description_str()
            );
        } throw shamrock_exc("cannot query single precision box the main field is not of f32_3 type");

        patch_data.sim_box.set_bounding_box<vectype>({bmin,bmax});

        logger::debug_ln("PatchScheduler", "box resized to :",
            bmin,bmax
        );

    }

    /**
     * @brief push data in the scheduler
     * The content of pdat as to be the same for each node
     * 
     * @param pdat the data to push
     */
    void allpush_data(shamrock::patch::PatchData & pdat);

    template<u32 dim>
    inline void make_patch_base_grid(std::array<u32,dim> patch_count){

        static_assert(dim == 3, "this is not implemented for dim != 3");

        u32 max_lin_patch_count = 0;
        for(u32 i = 0 ; i < dim; i++){
            max_lin_patch_count = sycl::max(max_lin_patch_count, patch_count[i]);
        }

        u64 coord_div_fact = shamrock::math::int_manip::get_next_pow2_val(max_lin_patch_count);

        u64 sz_root_patch = PatchScheduler::max_axis_patch_coord_lenght/coord_div_fact;

        
        std::vector<shamrock::patch::PatchCoord> coords;
        for(u32 x = 0; x < patch_count[0]; x++){
            for(u32 y = 0; y < patch_count[1]; y++){
                for(u32 z = 0; z < patch_count[2]; z++){
                    shamrock::patch::PatchCoord coord;

                    coord.x_min = sz_root_patch*(x);
                    coord.y_min = sz_root_patch*(y);
                    coord.z_min = sz_root_patch*(z);
                    coord.x_max = sz_root_patch*(x+1)-1;
                    coord.y_max = sz_root_patch*(y+1)-1;
                    coord.z_max = sz_root_patch*(z+1)-1;

                    coords.push_back(coord);
                }
            }
        }

        add_root_patches(coords);
    }

    /**
     * @brief modify the bounding box of the patch domain
     * 
     * @tparam vectype 
     * @param box 
     */
    template<class vectype>
    void set_coord_domain_bound(std::tuple<vectype, vectype> box){
        auto [a,b] = box;
        set_coord_domain_bound(a,b);
    }
    

    [[deprecated]]
    void dump_local_patches(std::string filename);

    std::vector<std::unique_ptr<shamrock::patch::PatchData>> gather_data(u32 rank);


    /**
     * @brief add patch to the scheduler
     *
     * //TODO find a better way to do this it is too error prone
     * 
     * @param p 
     * @param pdat 
     */
    //[[deprecated]]
    //inline u64 add_patch(shamrock::patch::Patch p, shamrock::patch::PatchData && pdat){
    //    p.id_patch = patch_list._next_patch_id;
    //    patch_list._next_patch_id ++;
//
    //    patch_list.global.push_back(p);
//
    //    patch_data.owned_data.insert({p.id_patch , pdat});
//
    //    return p.id_patch;
    //}

    void add_root_patch();

    [[deprecated]]
    void sync_build_LB(bool global_patch_sync, bool balance_load);



    /**
     * @brief for each macro for patchadata
     * exemple usage
     * ~~~~~{.cpp}
     *
     * sched.for_each_patch_data(
     *     [&](u64 id_patch, Patch cur_p, PatchData &pdat) {
     *          ....
     *     }
     * );
     *
     * ~~~~~
     *
     * @tparam Function The functor that will be used
     * @param fct 
     */
    template<class Function>
    inline void for_each_patch_data(Function && fct){

        for (auto &[id, pdat] : patch_data.owned_data) {

            if (! pdat.is_empty()) {

                shamrock::patch::Patch &cur_p = patch_list.global[patch_list.id_patch_to_global_idx[id]];

                if(!cur_p.is_err_mode()){
                    fct(id,cur_p,pdat);
                }

            }
        }

    }

    template<class Function>
    inline void for_each_patch(Function && fct){

        

        for (auto &[id, pdat] : patch_data.owned_data) {

            if (! pdat.is_empty()) {


                shamrock::patch::Patch &cur_p = patch_list.global[patch_list.id_patch_to_global_idx[id]];


                //TODO should feed the sycl queue to the lambda
                if(!cur_p.is_err_mode()){
                    fct(id,cur_p);
                }
            }
        }

    }

    //template<class Function, class Pfield>
    //inline void compute_patch_field(Pfield & field, MPI_Datatype & dtype , Function && lambda){
    //    field.local_nodes_value.resize(patch_list.local.size());
    //
    //    
    //
    //    for (u64 idx = 0; idx < patch_list.local.size(); idx++) {
    //
    //        Patch &cur_p = patch_list.local[idx];
    //
    //        PatchDataBuffer pdatbuf = attach_to_patchData(patch_data.owned_data.at(cur_p.id_patch));
    //
    //        field.local_nodes_value[idx] = lambda(shamsys::instance::get_compute_queue(),cur_p,pdatbuf);
    //
    //    }
    //
    //    field.build_global(dtype);
    //
    //}






    template<class Function, class Pfield>
    inline void compute_patch_field(Pfield & field, MPI_Datatype & dtype , Function && lambda){
        field.local_nodes_value.resize(patch_list.local.size());

        for (u64 idx = 0; idx < patch_list.local.size(); idx++) {

            shamrock::patch::Patch &cur_p = patch_list.local[idx];

            if(!cur_p.is_err_mode()){
                field.local_nodes_value[idx] = lambda(shamsys::instance::get_compute_queue(),cur_p,patch_data.owned_data.at(cur_p.id_patch));
            }
        }

        field.build_global(dtype);

    }

    /**
     * @brief add a root patch to the scheduler
     * 
     * @param coords coordinates of the patch
     * @return u64 the id of the made patch
     */
    std::vector<u64> add_root_patches(std::vector<shamrock::patch::PatchCoord> coords);

    shamrock::patch::SimulationBoxInfo & get_sim_box(){
        return patch_data.sim_box;
    }


    private:


    
    void split_patches(std::unordered_set<u64> split_rq);
    void merge_patches(std::unordered_set<u64> merge_rq);

    void set_patch_pack_values(std::unordered_set<u64> merge_rq);

};

inline void PatchScheduler::allpush_data(shamrock::patch::PatchData &pdat){

    for_each_patch_data(
        [&](u64 id_patch, shamrock::patch::Patch cur_p, shamrock::patch::PatchData &pdat_sched) {

            auto variant_main = pdl.get_main_field_any();

            std::visit([&](auto & arg){

                using base_t =
                            typename std::remove_reference<decltype(arg)>::type::field_T;

                if constexpr (shammath::sycl_utils::VectorProperties<base_t>::dimension == 3){
                    auto [bmin,bmax] = get_sim_box().partch_coord_to_domain<base_t>(cur_p)  ;

                    pdat_sched.insert_elements_in_range(pdat, bmin, bmax);
                }else{
                    throw std::runtime_error("this does not yet work with dimension different from 3");
                }

            }, variant_main);

        }
    );

}