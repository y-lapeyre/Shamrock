#pragma once

#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "scheduler/scheduler_patch_data.hpp"


class InterfaceVolumeGenerator{public:

    template<class vectype>
    static std::vector<PatchData> build_interface(sycl::queue & queue, PatchDataBuffer pdat_buf, std::vector<vectype> boxs_min, std::vector<vectype> boxs_max);

    
};


