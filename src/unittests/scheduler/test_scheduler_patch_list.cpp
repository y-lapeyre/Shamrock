#include "../shamrocktest.hpp"

#include "../../scheduler/scheduler_patch_list.hpp"
#include "test_patch_utils.hpp"


Test_start("SchedulerPatchList::",fake_patch_list_gen_test_volume,-1){

    std::vector<Patch> test_vec = make_fake_patch_list(200, 10);

    for(Patch & p : test_vec){

        u64 dx = p.x_max - p.x_min;
        u64 dy = p.y_max - p.y_min;
        u64 dz = p.z_max - p.z_min;

        Test_assert("is patch cube", dx == dy && dy == dz);
    }


}



Test_start("SchedulerPatchList::",build_select_corectness,-1){
    create_MPI_patch_type();

    //in the end this vector should be recovered in recv_vec
    std::vector<Patch> check_vec;
    //divide the check_vec in local_vector on each node
    std::vector<Patch> local_check_vec;
    make_global_local_check_vec(check_vec, local_check_vec);


    SchedulerPatchList patch_list = SchedulerPatchList();
    for(const Patch &p : check_vec){
        patch_list.global.push_back(p);
    }
    patch_list.build_local();


    //check corectness of local patch list
    bool corect_size = patch_list.local.size() == local_check_vec.size();
    Test_assert("corect size for local patch", corect_size);
    for(u32 i = 0 ; i < patch_list.local.size(); i++){
        if(corect_size){
            Test_assert("corect patch", patch_list.local[i] == local_check_vec[i]);
        }
    }


    patch_list.global.clear();
    patch_list.sync_global();


    corect_size = patch_list.global.size() == check_vec.size();
    Test_assert("corect size for global patch", corect_size);
    for(u32 i = 0 ; i < patch_list.global.size(); i++){
        if(corect_size){
            Test_assert("corect patch", patch_list.global[i] == check_vec[i]);
        }
    }



    free_MPI_patch_type();
}

