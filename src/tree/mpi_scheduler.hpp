
#include "patch.hpp"
#include <map>
#include <vector>


/**
 * @brief Patch containing the information associated with each patch not the actual data 
 * the data in packed to reduce data transferd through MPI.
 */
struct Patch{

    //patch information
    u64 id_patch;

    //patch tree info
    u64 id_parent;

    u64 id_child_r;
    u64 id_child_l;

    //Data
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;
    u32 data_count;

    ///////////////////////////////////////////////
    //patch tree evolution info
    ///////////////////////////////////////////////

    /**
     * @brief if true during next patch tree update child (and the PatchData associated) of this patch will be merged
     */
    bool should_merge_child;

    /**
     * @brief if true during next patch tree update this patch data will be split into two childs
     */
    bool should_split;

} __attribute__((packed));


//shared by all MPI node
inline std::vector<Patch> patch_table;



inline std::map<u64, PatchData*> local_patch_data;
inline std::vector<u64> local_patch_ids;