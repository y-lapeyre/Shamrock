#include "scheduler_mpi.hpp"



#include "../sys/sycl_handler.hpp"

#include "hilbertsfc.hpp"
#include "patch.hpp"








// TODO better parralelisation
std::vector<std::tuple<u64, i32, i32, i32>> make_change_list(std::vector<Patch> &global_patch_list, bool pack_tomerge_patch) {

    std::vector<std::tuple<u64, i32, i32, i32>> change_list;


    //generate hilbert code, load value, and index before sort

    // std::tuple<hilbert code ,load value ,index in global_patch_list>
    std::vector<std::tuple<u64, u64, u64>> patch_dt(global_patch_list.size());
    {

        cl::sycl::buffer<std::tuple<u64, u64, u64>> dt_buf(patch_dt);
        cl::sycl::buffer<Patch>                     patch_buf(global_patch_list);

        cl::sycl::range<1> range{global_patch_list.size()};

        host_queue->submit([&](cl::sycl::handler &cgh) {
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

        host_queue->submit([&](cl::sycl::handler &cgh) {
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



        host_queue->submit([&](cl::sycl::handler &cgh) {
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

            std::cout << i << " : " << old_owner << " -> " << new_owner << std::endl;

            if(new_owner != old_owner){
                change_list.push_back({i,old_owner,new_owner,tags_it_node[old_owner]});
                tags_it_node[old_owner] ++;
            }

            load_per_node[new_owner_table[i]] += global_patch_list[i].load_value;
            
        }

        for(i32 nid = 0 ; nid < mpi_handler::world_size; nid ++){
            std::cout << nid << " " << load_per_node[nid] << std::endl;
        }
        
    }



    return change_list;
}



void SchedulerMPI::sync_build_LB(bool global_patch_sync, bool balance_load, bool pack_tomerge_patch){

    if(global_patch_sync) patch_list.sync_global();

    if(balance_load){
        //real load balancing
        std::vector<std::tuple<u64, i32, i32,i32>> change_list = make_change_list(patch_list.global,pack_tomerge_patch);

        //exchange data
        patch_data.apply_change_list(change_list, patch_list);
    }

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}