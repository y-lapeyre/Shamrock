#pragma once

#include "CL/sycl/access/access.hpp"
#include "CL/sycl/accessor.hpp"
#include "CL/sycl/buffer.hpp"
#include "CL/sycl/builtins.hpp"
#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include <functional>

class InterfaceVolumeGenerator {
  public:
    template <class vectype>
    static std::vector<PatchData> build_interface(sycl::queue &queue, PatchDataBuffer pdat_buf,
                                                  std::vector<vectype> boxs_min, std::vector<vectype> boxs_max);
};

/**
 * @brief compact a buffer according to a mask
 *
 * @tparam T
 * @param buf_in buffer to compact
 * @param mask  after the algorithm data in mask will be garbage
 */
template <class T>
inline void array_compaction(sycl::queue &queue, sycl::buffer<T> &buf, const std::function<bool(T)> mask_func) {

    sycl::buffer<u64> remap_buf(buf.get_size());

    cl::sycl::range<1> range{buf.get_size()};

    queue.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor acc{buf, cgh, sycl::read_only};
        sycl::accessor mask{buf, cgh, sycl::write_only};

        cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
            u64 i = (u64)item.get_id(0);

            // u64_max if no, idx if true
            mask[i] = mask_func(acc[i]) ? i : u64_max;
        });
    });

    queue.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor acc{buf, cgh, sycl::read_write};
        sycl::accessor mask{buf, cgh, sycl::read_only};

        u64 len = buf.get_size();

        cgh.single_task([=]() {

        });
    });
}

template <class T> class OctreeMaxReducer {
  public:
    static T reduce(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {

        T tmp0 = sycl::max(v0, v1);
        T tmp1 = sycl::max(v2, v3);
        T tmp2 = sycl::max(v4, v5);
        T tmp3 = sycl::max(v6, v7);

        T tmpp0 = sycl::max(tmp0, tmp1);
        T tmpp1 = sycl::max(tmp2, tmp3);

        return sycl::max(tmpp0, tmpp1);
    }
};

template <class vectype>
class Interface_Generator {

    sycl::buffer<u64_2> interface_stack_buf;

    inline void gen_interfaces(SchedulerMPI &sched, SerialPatchTree<vectype> &sptree,
                               PatchField<typename vectype::element_type> pfield, vectype translate_factor,
                               vectype scale_factor) {

        const u64 local_pcount = sched.patch_list.local.size();
        sycl::buffer<u64> idx_patch(local_pcount);
        sycl::buffer<vectype> local_box_min_buf(local_pcount);
        sycl::buffer<vectype> local_box_max_buf(local_pcount);

        SyCLHandler & hndl = SyCLHandler::get_instance();

        PatchFieldReduction<typename vectype::element_type> pfield_reduced =
            sptree.template reduce_field<typename vectype::element_type, OctreeMaxReducer>(hndl.alt_queues[0], sched, pfield);
    
        
    
    }
};