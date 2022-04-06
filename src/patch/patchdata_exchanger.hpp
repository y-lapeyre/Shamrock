
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"


void patch_data_exchange_object(
    std::vector<Patch> & global_patch_list,
    std::vector<std::unique_ptr<PatchData>> &send_comm_pdat,
    std::vector<u64_2> &send_comm_vec,
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> & interface_map
    );