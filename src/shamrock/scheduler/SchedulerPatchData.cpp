// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file scheduler_patch_data.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Implementation of PatchData handling related function
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "SchedulerPatchData.hpp"

#include <stdexcept>
#include <vector>

#include "shamrock/legacy/io/logs.hpp"
#include "shamrock/legacy/patch/base/patchdata.hpp"

#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"
#include "shambase/string.hpp"
#include "shamrock/scheduler/HilbertLoadBalance.hpp"

//TODO use range based loop and emplace_back instead 

void SchedulerPatchData::apply_change_list(const shamrock::scheduler::LoadBalancingChangeList & change_list,SchedulerPatchList& patch_list){

    auto t = timings::start_timer("SchedulerPatchData::apply_change_list", timings::mpi);

    std::vector<PatchDataMpiRequest> rq_lst;

    using ChangeOp = shamrock::scheduler::LoadBalancingChangeList::ChangeOp;

    //send
    for(const ChangeOp op : change_list.change_ops){ // switch to range based
         //if i'm sender
        if(op.rank_owner_old == shamsys::instance::world_rank){
            auto & patchdata = owned_data.at(patch_list.global[op.patch_idx].id_patch);
            patchdata_isend(patchdata, rq_lst, op.rank_owner_new, op.tag_comm, MPI_COMM_WORLD);
        }
    }

    //receive
    for(const ChangeOp op : change_list.change_ops){
        auto & id_patch = patch_list.global[op.patch_idx].id_patch;
        
        //if i'm receiver
        if(op.rank_owner_new == shamsys::instance::world_rank){
            owned_data.emplace(id_patch,pdl);
            patchdata_irecv_probe(owned_data.at(id_patch), rq_lst, op.rank_owner_old , op.tag_comm, MPI_COMM_WORLD);
        }
    }

    waitall_pdat_mpi_rq(rq_lst);

    //erase old patchdata
    for(const ChangeOp op : change_list.change_ops){
        auto & id_patch = patch_list.global[op.patch_idx].id_patch;
        
        patch_list.global[op.patch_idx].node_owner_id = op.rank_owner_new;

        //if i'm sender delete old data
        if(op.rank_owner_new == shamsys::instance::world_rank){
            owned_data.erase(id_patch);
        }

    }

    t.stop();
}




template<class Vectype>
void split_patchdata(
    shamrock::patch::PatchData & original_pd,
    const shamrock::patch::SimulationBoxInfo & sim_box,
    const std::array<shamrock::patch::Patch, 8> patches,
    std::array<std::reference_wrapper<shamrock::patch::PatchData>,8> pdats){

    using ptype = typename shambase::sycl_utils::VectorProperties<Vectype>::component_type;

    auto [bmin_p0, bmax_p0] = sim_box.patch_coord_to_domain<Vectype>(patches[0]);
    auto [bmin_p1, bmax_p1] = sim_box.patch_coord_to_domain<Vectype>(patches[1]);
    auto [bmin_p2, bmax_p2] = sim_box.patch_coord_to_domain<Vectype>(patches[2]);
    auto [bmin_p3, bmax_p3] = sim_box.patch_coord_to_domain<Vectype>(patches[3]);
    auto [bmin_p4, bmax_p4] = sim_box.patch_coord_to_domain<Vectype>(patches[4]);
    auto [bmin_p5, bmax_p5] = sim_box.patch_coord_to_domain<Vectype>(patches[5]);
    auto [bmin_p6, bmax_p6] = sim_box.patch_coord_to_domain<Vectype>(patches[6]);
    auto [bmin_p7, bmax_p7] = sim_box.patch_coord_to_domain<Vectype>(patches[7]);

    original_pd.split_patchdata<Vectype>(pdats, 
        {bmin_p0, bmin_p1, bmin_p2, bmin_p3, bmin_p4, bmin_p5, bmin_p6, bmin_p7}, 
        {bmax_p0, bmax_p1, bmax_p2, bmax_p3, bmax_p4, bmax_p5, bmax_p6, bmax_p7});

}

template void split_patchdata<f32_3>(
    shamrock::patch::PatchData & original_pd,
    const shamrock::patch::SimulationBoxInfo & sim_box,
    const std::array<shamrock::patch::Patch, 8> patches,
    std::array<std::reference_wrapper<shamrock::patch::PatchData>,8> pdats);

template void split_patchdata<f64_3>(
    shamrock::patch::PatchData & original_pd,
    const shamrock::patch::SimulationBoxInfo & sim_box,
    const std::array<shamrock::patch::Patch, 8> patches,
    std::array<std::reference_wrapper<shamrock::patch::PatchData>,8> pdats);

template void split_patchdata<u32_3>(
    shamrock::patch::PatchData & original_pd,
    const shamrock::patch::SimulationBoxInfo & sim_box,
    const std::array<shamrock::patch::Patch, 8> patches,
    std::array<std::reference_wrapper<shamrock::patch::PatchData>,8> pdats);

template void split_patchdata<u64_3>(
    shamrock::patch::PatchData & original_pd,
    const shamrock::patch::SimulationBoxInfo & sim_box,
    const std::array<shamrock::patch::Patch, 8> patches,
    std::array<std::reference_wrapper<shamrock::patch::PatchData>,8> pdats);











void SchedulerPatchData::split_patchdata(u64 key_orginal, const std::array<shamrock::patch::Patch, 8> patches){

    
    auto search = owned_data.find(key_orginal);

    if (search != owned_data.end()) {

        shamrock::patch::PatchData & original_pd = search->second;

        shamrock::patch::PatchData pd0(pdl);
        shamrock::patch::PatchData pd1(pdl);
        shamrock::patch::PatchData pd2(pdl);
        shamrock::patch::PatchData pd3(pdl);
        shamrock::patch::PatchData pd4(pdl);
        shamrock::patch::PatchData pd5(pdl);
        shamrock::patch::PatchData pd6(pdl);
        shamrock::patch::PatchData pd7(pdl);

        if(pdl.check_main_field_type<f32_3>()){

            ::split_patchdata<f32_3>(
                    original_pd,
                    sim_box,
                    patches,
                    {pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7});
        }else if(pdl.check_main_field_type<f64_3>()){

            ::split_patchdata<f64_3>(
                    original_pd,
                    sim_box,
                    patches,
                    {pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7});
        }else if(pdl.check_main_field_type<u32_3>()){

            ::split_patchdata<u32_3>(
                    original_pd,
                    sim_box,
                    patches,
                    {pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7});
        }else if(pdl.check_main_field_type<u64_3>()){

            ::split_patchdata<u64_3>(
                    original_pd,
                    sim_box,
                    patches,
                    {pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7});
        }else{
            throw std::runtime_error("the main field does not match any");
        }

        owned_data.erase(key_orginal);

        owned_data.insert({patches[0].id_patch, pd0});
        owned_data.insert({patches[1].id_patch, pd1});
        owned_data.insert({patches[2].id_patch, pd2});
        owned_data.insert({patches[3].id_patch, pd3});
        owned_data.insert({patches[4].id_patch, pd4});
        owned_data.insert({patches[5].id_patch, pd5});
        owned_data.insert({patches[6].id_patch, pd6});
        owned_data.insert({patches[7].id_patch, pd7});
    }

}



void SchedulerPatchData::merge_patchdata(u64 new_key, u64 old_key0, u64 old_key1, u64 old_key2, u64 old_key3, u64 old_key4, u64 old_key5, u64 old_key6, u64 old_key7){

    auto search0 = owned_data.find(old_key0);
    auto search1 = owned_data.find(old_key1);
    auto search2 = owned_data.find(old_key2);
    auto search3 = owned_data.find(old_key3);
    auto search4 = owned_data.find(old_key4);
    auto search5 = owned_data.find(old_key5);
    auto search6 = owned_data.find(old_key6);
    auto search7 = owned_data.find(old_key7);

    if(search0 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key0));
    }
    if(search1 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key1));
    }
    if(search2 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key2));
    }
    if(search3 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key3));
    }
    if(search4 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key4));
    }
    if(search5 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key5));
    }
    if(search6 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key6));
    }
    if(search7 == owned_data.end()){
        throw shambase::throw_with_loc<std::runtime_error>(shambase::format_printf("patchdata for key=%d was not owned by the node",old_key7));
    }


    shamrock::patch::PatchData new_pdat(pdl);

    new_pdat.insert_elements(search0->second);
    new_pdat.insert_elements(search1->second);
    new_pdat.insert_elements(search2->second);
    new_pdat.insert_elements(search3->second);
    new_pdat.insert_elements(search4->second);
    new_pdat.insert_elements(search5->second);
    new_pdat.insert_elements(search6->second);
    new_pdat.insert_elements(search7->second);

    owned_data.erase(old_key0);
    owned_data.erase(old_key1);
    owned_data.erase(old_key2);
    owned_data.erase(old_key3);
    owned_data.erase(old_key4);
    owned_data.erase(old_key5);
    owned_data.erase(old_key6);
    owned_data.erase(old_key7);


    owned_data.insert({new_key ,new_pdat});

}