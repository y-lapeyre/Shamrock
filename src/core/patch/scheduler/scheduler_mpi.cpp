// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "scheduler_mpi.hpp"

#include <ctime>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/io/logs.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_field.hpp"
#include "shamsys/mpi_handler.hpp"
#include "loadbalancing_hilbert.hpp"

#include "core/patch/base/patchdata_layout.hpp"
#include "shamsys/sycl_handler.hpp"
#include "core/utils/time_utils.hpp"

#include "shamsys/sycl_mpi_interop.hpp"




//TODO move types init out
void PatchScheduler::init_mpi_required_types(){
    if(!is_mpi_sycl_interop_active()){
        create_sycl_mpi_types();
    }

    if(!patch::is_mpi_patch_type_active()){
        patch::create_MPI_patch_type();
    }
}

void PatchScheduler::free_mpi_required_types(){
    if(is_mpi_sycl_interop_active()){
        free_sycl_mpi_types();
    }

    if(patch::is_mpi_patch_type_active()){
        patch::free_MPI_patch_type();
    }
}

PatchScheduler::PatchScheduler(PatchDataLayout & pdl, u64 crit_split,u64 crit_merge) : pdl(pdl), patch_data(pdl){

    crit_patch_split = crit_split;
    crit_patch_merge = crit_merge;
    
}

PatchScheduler::~PatchScheduler(){

}

bool PatchScheduler::should_resize_box(bool node_in){
    u16 tmp = node_in;
    u16 out = 0;
    mpi::allreduce(&tmp, &out, 1, mpi_type_u16, MPI_MAX, MPI_COMM_WORLD);
    return out;
}




//TODO move Loadbalancing function to template state
void PatchScheduler::sync_build_LB(bool global_patch_sync, bool balance_load){

    if(global_patch_sync) patch_list.build_global();

    if(balance_load){
        //real load balancing
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = HilbertLB::make_change_list(patch_list.global);

        //exchange data
        patch_data.apply_change_list(change_list, patch_list);
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}

template<>
std::tuple<f32_3,f32_3> PatchScheduler::get_box_tranform(){
    if(pdl.xyz_mode == xyz64) throw shamrock_exc("cannot query single precision box, position is currently double precision");

    f32_3 translate_factor = patch_data.sim_box.min_box_sim_s;
    f32_3 scale_factor = (patch_data.sim_box.max_box_sim_s - patch_data.sim_box.min_box_sim_s)/HilbertLB::max_box_sz;

    return {translate_factor,scale_factor};
}

template<>
std::tuple<f64_3,f64_3> PatchScheduler::get_box_tranform(){
    if(pdl.xyz_mode == xyz32) throw shamrock_exc("cannot query double precision box, position is currently single precision");

    f64_3 translate_factor = patch_data.sim_box.min_box_sim_d;
    f64_3 scale_factor = (patch_data.sim_box.max_box_sim_d - patch_data.sim_box.min_box_sim_d)/HilbertLB::max_box_sz;

    return {translate_factor,scale_factor};
}


template<>
std::tuple<f32_3,f32_3> PatchScheduler::get_box_volume(){
   if(pdl.xyz_mode == xyz64) throw shamrock_exc("cannot query single precision box, position is currently double precision");

    return {patch_data.sim_box.min_box_sim_s,patch_data.sim_box.max_box_sim_s};
}

template<>
std::tuple<f64_3,f64_3> PatchScheduler::get_box_volume(){
    if(pdl.xyz_mode == xyz32) throw shamrock_exc("cannot query double precision box, position is currently single precision");

    return {patch_data.sim_box.min_box_sim_d,patch_data.sim_box.max_box_sim_d};
}

template<>
void PatchScheduler::set_box_volume(std::tuple<f32_3,f32_3> box){
    if(pdl.xyz_mode == xyz64) throw shamrock_exc("cannot query single precision box, position is currently double precision");

    patch_data.sim_box.min_box_sim_s = std::get<0>(box);
    patch_data.sim_box.max_box_sim_s = std::get<1>(box);

    logger::debug_ln("PatchScheduler", "box resized to :",
        patch_data.sim_box.min_box_sim_s,
        patch_data.sim_box.max_box_sim_s 
    );

}

template<>
void PatchScheduler::set_box_volume(std::tuple<f64_3,f64_3> box){
    if(pdl.xyz_mode == xyz32) throw shamrock_exc("cannot query double precision box, position is currently single precision");

    patch_data.sim_box.min_box_sim_d = std::get<0>(box);
    patch_data.sim_box.max_box_sim_d = std::get<1>(box);

    logger::debug_ln("PatchScheduler", "box resized to :",
        patch_data.sim_box.min_box_sim_d,
        patch_data.sim_box.max_box_sim_d 
    );

}



//TODO clean the output of this function
void PatchScheduler::scheduler_step(bool do_split_merge, bool do_load_balancing){

    //std::cout << dump_status();

    auto global_timer = timings::start_timer("SchedulerMPI::scheduler_step", timings::function);

    if(!is_mpi_sycl_interop_active()) throw shamrock_exc("sycl mpi interop not initialized");
    if(!patch::is_mpi_patch_type_active()) throw shamrock_exc("mpi patch type not initialized");

    Timer timer;

    std::cout << " -> running scheduler step\n";

    //std::cout << "sync global" <<std::endl;

    
    timer.start();
    patch_list.build_global();
    timer.end();
    std::cout << " | sync global : " << timer.get_time_str() << std::endl;

    //std::cout << dump_status();

    std::unordered_set<u64> split_rq;
    std::unordered_set<u64> merge_rq;

    if(do_split_merge){
        //std::cout << dump_status() << std::endl;

        //std::cout << "build_global_idx_map" <<std::endl;
        
        timer.start();//TODO check if it it used outside of split merge -> maybe need to be put before the if
        patch_list.build_global_idx_map();
        timer.end();
        std::cout << " | build_global_idx_map : " << timer.get_time_str() << std::endl;

        //std::cout << dump_status() << std::endl;



        //std::cout << "tree partial_values_reduction" <<std::endl;
        timer.start();
        patch_tree.partial_values_reduction(
                patch_list.global, 
                patch_list.id_patch_to_global_idx);
        timer.end();
        std::cout << " | partial_values_reduction : " << timer.get_time_str() << std::endl;


        //std::cout << dump_status() << std::endl;

        // Generate merge and split request  
        timer.start();
        split_rq = patch_tree.get_split_request(crit_patch_split);
        merge_rq = patch_tree.get_merge_request(crit_patch_merge);
        timer.end();
        std::cout << " | gen split/merge op : " << timer.get_time_str() << std::endl;



        std::cout <<        "   | ---- patch operation requests ---- \n";
        std::cout << format("      split : %-6d   | merge : %-6d",split_rq.size(), merge_rq.size()) << std::endl;

        /*
        std::cout << "     |-> split rq : ";
        for(u64 i : split_rq){
            std::cout << i << " ";
        }std::cout << std::endl;
        //*/

        /*
        std::cout << "     |-> merge rq : ";
        for(u64 i : merge_rq){
            std::cout << i << " ";
        }std::cout << std::endl;
        //*/

        //std::cout << dump_status() << std::endl;

        //std::cout << "split_patches" <<std::endl;
        timer.start();
        split_patches(split_rq);
        timer.end();
        std::cout << " | apply splits : " << timer.get_time_str() << std::endl;

        //std::cout << dump_status() << std::endl;

        //check not necessary if no splits
        timer.start();
        patch_list.build_global_idx_map();
        timer.end();
        std::cout << " | build_global_idx_map : " << timer.get_time_str() << std::endl;



        timer.start();
        set_patch_pack_values(merge_rq);
        timer.end();
        std::cout << " | set_patch_pack_values : " << timer.get_time_str() << std::endl;
    }


    if(do_load_balancing){
        auto t = timings::start_timer("load balancing", timings::function);
        timer.start();
        // generate LB change list 
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = 
            HilbertLB::make_change_list(patch_list.global);
        timer.end();
        std::cout << " | load balancing gen op : " << timer.get_time_str() << std::endl;

        std::cout <<        "   | ---- load balancing ---- \n";
        std::cout << format("      move op : %-6d",change_list.size()) << std::endl;

        timer.start();
        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
        timer.end();
        std::cout << " | apply balancing : " << timer.get_time_str() << std::endl;
        t.stop();
    }

    //std::cout << dump_status();


    if(do_split_merge){
        patch_list.build_local_idx_map();
        merge_patches(merge_rq);
    }



    //TODO should be moved out of the scheduler step
    owned_patch_id = patch_list.build_local();
    patch_list.reset_local_pack_index();
    patch_list.build_local_idx_map();
    patch_list.build_global_idx_map();//TODO check if required : added because possible bug because of for each patch & serial patch tree
    update_local_dtcnt_value();
    update_local_load_value();

    global_timer.stop();

    //std::cout << dump_status();

}

/*
void SchedulerMPI::scheduler_step(bool do_split_merge,bool do_load_balancing){

    // update patch list  
    patch_list.sync_global();


    if(do_split_merge){
        // rebuild patch index map
        patch_list.build_global_idx_map();

        // apply reduction on leafs and corresponding parents
        patch_tree.partial_values_reduction(
            patch_list.global, 
            patch_list.id_patch_to_global_idx);

        // Generate merge and split request  
        std::unordered_set<u64> split_rq = patch_tree.get_split_request(crit_patch_split);
        std::unordered_set<u64> merge_rq = patch_tree.get_merge_request(crit_patch_merge);
        

        // apply split requests
        // update patch_list.global same on every node 
        // and split patchdata accordingly if owned
        // & update tree
        split_patches(split_rq);

        // update packing index 
        // same operation on evey cluster nodes
        set_patch_pack_values(merge_rq);

        // update patch list
        // necessary to update load values in splitted patches
        // alternative : disable this step and set fake load values (load parent / 8)
        //alternative impossible if gravity because we have to compute the multipole
        owned_patch_id = patch_list.build_local();
        patch_list.sync_global();
    }

    if(do_load_balancing){
        // generate LB change list 
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = 
            make_change_list(patch_list.global);

        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
    }

    if(do_split_merge){
        // apply merge requests  
        // & update tree
        merge_patches(merge_rq);



        // if(Merge) update patch list  
        if(! merge_rq.empty()){
            owned_patch_id = patch_list.build_local();
            patch_list.sync_global();
        }
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}
//*/






std::string PatchScheduler::dump_status(){

    std::stringstream ss;

    ss << "----- MPI Scheduler dump -----\n\n";
    ss << " -> SchedulerPatchList\n";

    ss << "    len global : " << patch_list.global.size()<<"\n";
    ss << "    len local  : " << patch_list.local.size()<<"\n";

    ss << "    global content : \n";
    for (Patch & p : patch_list.global) {

        ss << "      -> " 
            << p.id_patch << " : " 
            << p.data_count << " "
            << p.load_value << " "
            << p.node_owner_id << " "
            << p.pack_node_index << " "
            << "( ["<< p.x_min << "," << p.x_max << "] "
            << " ["<< p.y_min << "," << p.y_max << "] "
            << " ["<< p.z_min << "," << p.z_max << "] )\n";

    }
    ss << "    local content : \n";
    for (Patch & p : patch_list.local) {

        ss << "      -> id : " 
            << p.id_patch << " : " 
            << p.data_count << " "
            << p.load_value << " "
            << p.node_owner_id << " "
            << p.pack_node_index << " "
            << "( ["<< p.x_min << "," << p.x_max << "] "
            << " ["<< p.y_min << "," << p.y_max << "] "
            << " ["<< p.z_min << "," << p.z_max << "] )\n";
            
    }


    ss << " -> SchedulerPatchData\n";
    ss << "    owned data : \n";

    for (auto & [pid,pdat] : patch_data.owned_data) {
        ss << "patch id : " << pid << " len = " << pdat.get_obj_cnt() << "\n";
    }

    /*
    for(auto & [k,pdat] : patch_data.owned_data){
        ss << "      -> id : " << k << " len : (" << 
            pdat.pos_s.size() << " " <<pdat.pos_d.size() << " " <<
            pdat.U1_s.size() << " " <<pdat.U1_d.size() << " " <<
            pdat.U3_s.size() << " " <<pdat.U3_d.size() << " " 
        << ")\n";
    }
    */


    ss << " -> SchedulerPatchTree\n";

    for(auto & [k,pnode] : patch_tree.tree){
        ss << format("      -> id : %d  -> (%d %d %d %d %d %d %d %d) <=> %d\n",
        k,
        pnode.childs_id[0],
        pnode.childs_id[1],
        pnode.childs_id[2],
        pnode.childs_id[3],
        pnode.childs_id[4],
        pnode.childs_id[5],
        pnode.childs_id[6],
        pnode.childs_id[7],
         pnode.linked_patchid);
    }



    return ss.str();

}



inline void PatchScheduler::split_patches(std::unordered_set<u64> split_rq){
    auto t = timings::start_timer("SchedulerMPI::split_patches", timings::function);
    for(u64 tree_id : split_rq){

        patch_tree.split_node(tree_id);
        PatchTree::PTNode & splitted_node = patch_tree.tree[tree_id];

        auto [idx_p0,idx_p1,idx_p2,idx_p3,idx_p4,idx_p5,idx_p6,idx_p7] 
            =  patch_list.split_patch(splitted_node.linked_patchid);

        u64 old_patch_id = splitted_node.linked_patchid;

        splitted_node.linked_patchid = u64_max;
        patch_tree.tree[splitted_node.childs_id[0]].linked_patchid = patch_list.global[idx_p0].id_patch;
        patch_tree.tree[splitted_node.childs_id[1]].linked_patchid = patch_list.global[idx_p1].id_patch;
        patch_tree.tree[splitted_node.childs_id[2]].linked_patchid = patch_list.global[idx_p2].id_patch;
        patch_tree.tree[splitted_node.childs_id[3]].linked_patchid = patch_list.global[idx_p3].id_patch;
        patch_tree.tree[splitted_node.childs_id[4]].linked_patchid = patch_list.global[idx_p4].id_patch;
        patch_tree.tree[splitted_node.childs_id[5]].linked_patchid = patch_list.global[idx_p5].id_patch;
        patch_tree.tree[splitted_node.childs_id[6]].linked_patchid = patch_list.global[idx_p6].id_patch;
        patch_tree.tree[splitted_node.childs_id[7]].linked_patchid = patch_list.global[idx_p7].id_patch;

        patch_data.split_patchdata(
            old_patch_id,
            patch_list.global[idx_p0], 
            patch_list.global[idx_p1],
            patch_list.global[idx_p2],
            patch_list.global[idx_p3],
            patch_list.global[idx_p4],
            patch_list.global[idx_p5],
            patch_list.global[idx_p6],
            patch_list.global[idx_p7]);

    }
    t.stop();
}

inline void PatchScheduler::merge_patches(std::unordered_set<u64> merge_rq){
    auto t = timings::start_timer("SchedulerMPI::merge_patches", timings::function);
    for(u64 tree_id : merge_rq){

        PatchTree::PTNode & to_merge_node = patch_tree.tree[tree_id];

        std::cout << "merging patch tree id : " << tree_id << "\n";
        

        u64 patch_id0 = patch_tree.tree[to_merge_node.childs_id[0]].linked_patchid;
        u64 patch_id1 = patch_tree.tree[to_merge_node.childs_id[1]].linked_patchid;
        u64 patch_id2 = patch_tree.tree[to_merge_node.childs_id[2]].linked_patchid;
        u64 patch_id3 = patch_tree.tree[to_merge_node.childs_id[3]].linked_patchid;
        u64 patch_id4 = patch_tree.tree[to_merge_node.childs_id[4]].linked_patchid;
        u64 patch_id5 = patch_tree.tree[to_merge_node.childs_id[5]].linked_patchid;
        u64 patch_id6 = patch_tree.tree[to_merge_node.childs_id[6]].linked_patchid;
        u64 patch_id7 = patch_tree.tree[to_merge_node.childs_id[7]].linked_patchid;
        
        //print list of patch that will merge
        //std::cout << format("  -> (%d %d %d %d %d %d %d %d)\n", patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);
        
        if(patch_list.global[patch_list.id_patch_to_global_idx[ patch_id0 ]].node_owner_id == mpi_handler::world_rank){
            patch_data.merge_patchdata(patch_id0, patch_id0, patch_id1, patch_id2, patch_id3, patch_id4, patch_id5, patch_id6, patch_id7);
        }

        patch_list.merge_patch(
            patch_list.id_patch_to_global_idx[ patch_id0 ],
            patch_list.id_patch_to_global_idx[ patch_id1 ],
            patch_list.id_patch_to_global_idx[ patch_id2 ],
            patch_list.id_patch_to_global_idx[ patch_id3 ],
            patch_list.id_patch_to_global_idx[ patch_id4 ],
            patch_list.id_patch_to_global_idx[ patch_id5 ],
            patch_list.id_patch_to_global_idx[ patch_id6 ],
            patch_list.id_patch_to_global_idx[ patch_id7 ]);

        patch_tree.merge_node_dm1(tree_id);

        to_merge_node.linked_patchid = patch_id0;

    }
    t.stop();
}


inline void PatchScheduler::set_patch_pack_values(std::unordered_set<u64> merge_rq){

    for(u64 tree_id : merge_rq){

        PatchTree::PTNode & to_merge_node = patch_tree.tree[tree_id];

        u64 idx_pack = patch_list.id_patch_to_global_idx[
            patch_tree.tree[to_merge_node.childs_id[0]].linked_patchid
            ];

        //std::cout << "node id : " << patch_list.global[idx_pack].id_patch << " should merge with : ";

        for (u8 i = 1; i < 8; i++) {
            //std::cout <<  patch_tree.tree[to_merge_node.childs_id[i]].linked_patchid << " ";
            patch_list.global[
                patch_list.id_patch_to_global_idx[
                        patch_tree.tree[to_merge_node.childs_id[i]].linked_patchid
                    ]
                ].pack_node_index = idx_pack;
        }//std::cout << std::endl;

    }

}











void PatchScheduler::dump_local_patches(std::string filename){
    std::ofstream fout(filename);

    if(pdl.xyz_mode == xyz32){

        std::tuple<f32_3,f32_3> box_transform = get_box_tranform<f32_3>();

        for(const Patch & p : patch_list.local){
            
            f32_3 box_min = f32_3{p.x_min, p.y_min,
                                    p.z_min} *
                                std::get<1>(box_transform) +
                            std::get<0>(box_transform);
            f32_3 box_max = (f32_3{p.x_max, p.y_max,
                                    p.z_max} +
                            1) *
                                std::get<1>(box_transform) +
                            std::get<0>(box_transform);


            fout << 
            p.id_patch << "|" << 
            p.data_count << "|" << 
            p.load_value << "|" << 
            p.node_owner_id << "|" << 
            p.pack_node_index << "|" << 
            box_min.x() << "|" << 
            box_max.x() << "|" << 
            box_min.y() << "|" << 
            box_max.y() << "|" << 
            box_min.z() << "|" << 
            box_max.z() << "|" << "\n";
        }

        fout.close();

    }else if (pdl.xyz_mode == xyz64){
        
        std::tuple<f64_3,f64_3> box_transform = get_box_tranform<f64_3>();

        for(const Patch & p : patch_list.local){
            
            f64_3 box_min = f64_3{p.x_min, p.y_min,
                                    p.z_min} *
                                std::get<1>(box_transform) +
                            std::get<0>(box_transform);
            f64_3 box_max = (f64_3{p.x_max, p.y_max,
                                    p.z_max} +
                            1) *
                                std::get<1>(box_transform) +
                            std::get<0>(box_transform);


            fout << 
            p.id_patch << "|" << 
            p.data_count << "|" << 
            p.load_value << "|" << 
            p.node_owner_id << "|" << 
            p.pack_node_index << "|" << 
            box_min.x() << "|" << 
            box_max.x() << "|" << 
            box_min.y() << "|" << 
            box_max.y() << "|" << 
            box_min.z() << "|" << 
            box_max.z() << "|" << "\n";
        }

        fout.close();

    }else{
        throw shamrock_exc("position precision was not set");
    }
}




std::vector<std::unique_ptr<PatchData>> PatchScheduler::gather_data(u32 rank){

    auto plist = this->patch_list.global;
    auto pdata = this->patch_data.owned_data;

    std::vector<std::unique_ptr<PatchData>> ret;


    if (mpi_handler::world_rank == 0) {
         ret.resize(plist.size());
    }

    

    std::vector<PatchDataMpiRequest> rq_lst;

    for (u32 i = 0; i < plist.size(); i++) {
        auto & cpatch = plist[i];
        if(cpatch.node_owner_id == mpi_handler::world_rank){
            patchdata_isend(pdata.at(cpatch.id_patch), rq_lst, 0, i, MPI_COMM_WORLD);
        }
    }

    if(mpi_handler::world_rank == 0){
        for (u32 i = 0; i < plist.size(); i++) {
            ret.at(i) = std::make_unique<PatchData>(pdl);
            patchdata_irecv_probe(*ret.at(i),rq_lst, plist[i].node_owner_id, i, MPI_COMM_WORLD);
        }
    }


    waitall_pdat_mpi_rq(rq_lst);

    return ret;

}