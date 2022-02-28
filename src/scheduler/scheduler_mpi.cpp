#include "scheduler_mpi.hpp"



#include "../sys/sycl_handler.hpp"

#include "hilbertsfc.hpp"
#include "patch.hpp"
#include <unordered_set>



#include "../utils/time_utils.hpp"




// TODO better parralelisation
std::vector<std::tuple<u64, i32, i32, i32>> make_change_list(std::vector<Patch> &global_patch_list) {

    std::vector<std::tuple<u64, i32, i32, i32>> change_list;


    //generate hilbert code, load value, and index before sort

    // std::tuple<hilbert code ,load value ,index in global_patch_list>
    std::vector<std::tuple<u64, u64, u64>> patch_dt(global_patch_list.size());
    {

        cl::sycl::buffer<std::tuple<u64, u64, u64>> dt_buf(patch_dt);
        cl::sycl::buffer<Patch>                     patch_buf(global_patch_list);

        cl::sycl::range<1> range{global_patch_list.size()};

        SyCLHandler::get_instance().alt_queues[0].submit([&](cl::sycl::handler &cgh) {
            auto ptch = patch_buf.get_access<sycl::access::mode::read>(cgh);
            auto pdt  = dt_buf.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class Compute_HilbLoad>(range, [=](cl::sycl::item<1> item) {
                u64 i = (u64)item.get_id(0);

                Patch p = ptch[i];

                pdt[i] = {compute_hilbert_index<21>(p.x_min, p.y_min, p.z_min), p.load_value, i};
            });
        });

    }


    //sort hilbert code
    std::sort(patch_dt.begin(), patch_dt.end());


    //compute increments for load
    for (u64 i = 1; i < global_patch_list.size(); i++) {
        std::get<1>(patch_dt[i]) += std::get<1>(patch_dt[i-1]);
    }

    /*
    {
        double target_datacnt = double(std::get<1>(patch_dt[global_patch_list.size()-1]))/mpi_handler::world_size;
        for(auto t : patch_dt){
            std::cout <<
                std::get<0>(t) << " "<<
                std::get<1>(t) << " "<<
                std::get<2>(t) << " "<<
                sycl::clamp(
                    i32(std::get<1>(t)/target_datacnt)
                    ,0,mpi_handler::world_size-1) << " " << (std::get<1>(t)/target_datacnt) << 
                std::endl;
        }
    }
    */


    //compute new owners
    std::vector<i32> new_owner_table(global_patch_list.size());
    {

        cl::sycl::buffer<std::tuple<u64, u64, u64>> dt_buf(patch_dt);
        cl::sycl::buffer<i32> new_owner(new_owner_table);
        cl::sycl::buffer<Patch>                     patch_buf(global_patch_list);

        cl::sycl::range<1> range{global_patch_list.size()};

        SyCLHandler::get_instance().alt_queues[0].submit([&](cl::sycl::handler &cgh) {
            auto pdt  = dt_buf.get_access<sycl::access::mode::read>(cgh);
            auto chosen_node = new_owner.get_access<sycl::access::mode::discard_write>(cgh);

            //TODO [potential issue] here must check that the conversion to double doesn't mess up the target dt_cnt or find another way
            double target_datacnt = double(std::get<1>(patch_dt[global_patch_list.size()-1]))/mpi_handler::world_size;

            i32 wsize = mpi_handler::world_size;


            cgh.parallel_for<class Write_chosen_node>(range, [=](cl::sycl::item<1> item) {
                u64 i = (u64)item.get_id(0);

                u64 id_ptable = std::get<2>(pdt[i]);

                chosen_node[id_ptable] = sycl::clamp(
                    i32(std::get<1>(pdt[i])/target_datacnt)
                    ,0,wsize-1);

            });
        });


        //pack nodes
        SyCLHandler::get_instance().alt_queues[0].submit([&](cl::sycl::handler &cgh) {
            auto ptch = patch_buf.get_access<sycl::access::mode::read>(cgh);
            auto pdt  = dt_buf.get_access<sycl::access::mode::read>(cgh);
            auto chosen_node = new_owner.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class Edit_chosen_node>(range, [=](cl::sycl::item<1> item) {
                u64 i = (u64)item.get_id(0);


                if(ptch[i].pack_node_index != u64_max){
                    chosen_node[i] = chosen_node[ptch[i].pack_node_index];
                }

            });
        });

    }




    //make change list
    {   
        std::vector<u64> load_per_node(mpi_handler::world_size);

        std::vector<i32> tags_it_node(mpi_handler::world_size);
        for(u64 i = 0 ; i < global_patch_list.size(); i++){

            i32 old_owner = global_patch_list[i].node_owner_id;
            i32 new_owner = new_owner_table[i];

            // TODO add bool for optional print verbosity
            //std::cout << i << " : " << old_owner << " -> " << new_owner << std::endl;

            if(new_owner != old_owner){
                change_list.push_back({i,old_owner,new_owner,tags_it_node[old_owner]});
                tags_it_node[old_owner] ++;
            }

            load_per_node[new_owner_table[i]] += global_patch_list[i].load_value;
            
        }

        std::cout << "load after balancing" << std::endl;
        for(i32 nid = 0 ; nid < mpi_handler::world_size; nid ++){
            std::cout << nid << " " << load_per_node[nid] << std::endl;
        }
        
    }



    return change_list;
}



void SchedulerMPI::sync_build_LB(bool global_patch_sync, bool balance_load){

    if(global_patch_sync) patch_list.sync_global();

    if(balance_load){
        //real load balancing
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = make_change_list(patch_list.global);

        //exchange data
        patch_data.apply_change_list(change_list, patch_list);
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}

//TODO clean the output of this function
void SchedulerMPI::scheduler_step(bool do_split_merge, bool do_load_balancing){
    Timer timer;

    std::cout << " -> running scheduler step\n";

    //std::cout << "sync global" <<std::endl;

    
    timer.start();
    patch_list.sync_global();
    timer.end();
    std::cout << " | sync global : " << timer.get_time_str() << std::endl;



    std::unordered_set<u64> split_rq;
    std::unordered_set<u64> merge_rq;

    if(do_split_merge){
        //std::cout << dump_status() << std::endl;

        //std::cout << "build_global_idx_map" <<std::endl;
        timer.start();
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
        timer.start();
        // generate LB change list 
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = 
            make_change_list(patch_list.global);
        timer.end();
        std::cout << " | load balancing gen op : " << timer.get_time_str() << std::endl;

        std::cout <<        "   | ---- load balancing ---- \n";
        std::cout << format("      move op : %-6d",change_list.size()) << std::endl;

        timer.start();
        // apply LB change list
        patch_data.apply_change_list(change_list, patch_list);
        timer.end();
        std::cout << " | apply balancing : " << timer.get_time_str() << std::endl;
    }


    if(do_split_merge){
        patch_list.build_local_idx_map();
        merge_patches(merge_rq);
    }



    //TODO should be moved out of the scheduler step
    owned_patch_id = patch_list.build_local();
    patch_list.reset_local_pack_index();
    patch_list.build_local_idx_map();
    update_local_dtcnt_value();
    update_local_load_value();

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






std::string SchedulerMPI::dump_status(){

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

    for(auto & [k,pdat] : patch_data.owned_data){
        ss << "      -> id : " << k << " len : (" << 
            pdat.pos_s.size() << " " <<pdat.pos_d.size() << " " <<
            pdat.U1_s.size() << " " <<pdat.U1_d.size() << " " <<
            pdat.U3_s.size() << " " <<pdat.U3_d.size() << " " 
        << ")\n";
    }


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



inline void SchedulerMPI::split_patches(std::unordered_set<u64> split_rq){
    for(u64 tree_id : split_rq){

        patch_tree.split_node(tree_id);
        PTNode & splitted_node = patch_tree.tree[tree_id];

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
}

inline void SchedulerMPI::merge_patches(std::unordered_set<u64> merge_rq){
    for(u64 tree_id : merge_rq){

        PTNode & to_merge_node = patch_tree.tree[tree_id];

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
}


inline void SchedulerMPI::set_patch_pack_values(std::unordered_set<u64> merge_rq){

    for(u64 tree_id : merge_rq){

        PTNode & to_merge_node = patch_tree.tree[tree_id];

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