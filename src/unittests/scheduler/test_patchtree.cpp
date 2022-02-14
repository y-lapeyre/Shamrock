#include "../../scheduler/patchtree.hpp"

#include "../shamrocktest.hpp"
#include <random>
#include <vector>
#include "../../scheduler/scheduler_patch_list.hpp"


Test_start("", testpatchtree, 1){

    std::vector<Patch> global = make_fake_patch_list(20,10);



    PatchTree pt;

    pt.build_from_patchtable(global);



}