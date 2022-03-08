#pragma once

#include "CL/sycl/access/access.hpp"
#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patch/serialpatchtree.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include "utils/geometry_utils.hpp"
#include "utils/string_utils.hpp"
#include <fstream>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

class InterfaceVolumeGenerator {
  public:
    template <class vectype>
    static std::vector<PatchData> build_interface(sycl::queue &queue, PatchDataBuffer pdat_buf,
                                                  std::vector<vectype> boxs_min, std::vector<vectype> boxs_max);
};

template <class vectype, class field_type, class InterfaceSelector> class Interface_Generator {

  private:


    struct InterfaceCom{
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;
        vectype interf_box_min;
        vectype interf_box_max;
    };

    inline static sycl::buffer<InterfaceCom, 2> get_interface_list_v1(SchedulerMPI &sched, SerialPatchTree<vectype> &sptree,
                                             PatchField<typename vectype::element_type> pfield) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        const u64 local_pcount  = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        if (local_pcount == 0)
            throw std::runtime_error("local patch count is zero this function can not run");

        sycl::buffer<u64> patch_ids_buf(local_pcount);
        sycl::buffer<vectype> local_box_min_buf(local_pcount);
        sycl::buffer<vectype> local_box_max_buf(local_pcount);

        sycl::buffer<vectype> global_box_min_buf(global_pcount);
        sycl::buffer<vectype> global_box_max_buf(global_pcount);

        sycl::buffer<InterfaceCom, 2> interface_list_buf({local_pcount, global_pcount});
        sycl::buffer<u64> global_ids_buf(global_pcount);

        {
            auto pid      = patch_ids_buf.get_access<sycl::access::mode::discard_write>();
            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::discard_write>();
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::discard_write>();

            auto gbox_min = global_box_min_buf.template get_access<sycl::access::mode::discard_write>();
            auto gbox_max = global_box_max_buf.template get_access<sycl::access::mode::discard_write>();

            std::tuple<vectype, vectype> box_transform = sched.get_box_tranform<vectype>();

            for (u64 i = 0; i < local_pcount; i++) {
                pid[i] = sched.patch_list.local[i].id_patch;

                lbox_min[i] = vectype{sched.patch_list.local[i].x_min, sched.patch_list.local[i].y_min,
                                      sched.patch_list.local[i].z_min} *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform);
                lbox_max[i] = (vectype{sched.patch_list.local[i].x_max, sched.patch_list.local[i].y_max,
                                       sched.patch_list.local[i].z_max} +
                               1) *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform);
            }

            auto g_pid = global_ids_buf.get_access<sycl::access::mode::discard_write>();
            for (u64 i = 0; i < global_pcount; i++) {
                g_pid[i] = sched.patch_list.global[i].id_patch;

                gbox_min[i] = vectype{sched.patch_list.global[i].x_min, sched.patch_list.global[i].y_min,
                                      sched.patch_list.global[i].z_min} *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform);
                gbox_max[i] = (vectype{sched.patch_list.global[i].x_max, sched.patch_list.global[i].y_max,
                                       sched.patch_list.global[i].z_max} +
                               1) *
                                  std::get<1>(box_transform) +
                              std::get<0>(box_transform);
            }
        }

        // PatchFieldReduction<field_type> pfield_reduced = sptree.template reduce_field<field_type,
        // OctreeMaxReducer>(hndl.alt_queues[0], sched, pfield);

        sycl::buffer<field_type> buf_local_field_val(pfield.local_nodes_value);
        sycl::buffer<field_type> buf_global_field_val(pfield.global_values);

        hndl.alt_queues[0].submit([&](cl::sycl::handler &cgh) {
            auto pid  = patch_ids_buf.get_access<sycl::access::mode::read>(cgh);
            auto gpid = global_ids_buf.get_access<sycl::access::mode::read>(cgh);

            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read>(cgh);
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read>(cgh);

            auto gbox_min = global_box_min_buf.template get_access<sycl::access::mode::read>(cgh);
            auto gbox_max = global_box_max_buf.template get_access<sycl::access::mode::read>(cgh);

            auto local_field  = buf_local_field_val.template get_access<sycl::access::mode::read>(cgh);
            auto global_field = buf_global_field_val.template get_access<sycl::access::mode::read>(cgh);

            auto interface_list = interface_list_buf.template get_access<sycl::access::mode::discard_write>(cgh);

            u64 cnt_patch = global_pcount;

            cgh.parallel_for(sycl::range<1>(local_pcount), [=](cl::sycl::item<1> item) {
                u64 cur_patch_idx    = (u64)item.get_id(0);
                u64 cur_patch_id     = pid[cur_patch_idx];
                vectype cur_lbox_min = lbox_min[cur_patch_idx];
                vectype cur_lbox_max = lbox_max[cur_patch_idx];

                u64 interface_ptr = 0;

                for (u64 test_patch_idx = 0; test_patch_idx < cnt_patch; test_patch_idx++) {

                    vectype test_lbox_min = gbox_min[test_patch_idx];
                    vectype test_lbox_max = gbox_max[test_patch_idx];
                    u64 test_patch_id     = gpid[test_patch_idx];

                    {
                        std::tuple<vectype, vectype> b1 = InterfaceSelector::get_neighbourg_box_sz(
                            cur_lbox_min, cur_lbox_max, global_field[test_patch_idx], local_field[cur_patch_idx]);
                        std::tuple<vectype, vectype> b2 = InterfaceSelector::get_compute_box_sz(
                            test_lbox_min, test_lbox_max, global_field[test_patch_idx], local_field[cur_patch_idx]);

                        if (BBAA::intersect_not_null_cella_b(std::get<0>(b1), std::get<1>(b1), std::get<0>(b2),
                                                             std::get<1>(b2)) &&
                            (test_patch_id != cur_patch_id)) {

                            std::tuple<vectype,vectype> box_interf = BBAA::get_intersect_cella_b(std::get<0>(b1), std::get<1>(b1), std::get<0>(b2),
                                                             std::get<1>(b2));
                            interface_list[{cur_patch_idx, interface_ptr}] = InterfaceCom{
                                cur_patch_idx, 
                                test_patch_idx,
                                cur_patch_id,
                                test_patch_id,
                                std::get<0>(box_interf),
                                std::get<1>(box_interf)
                                };
                            interface_ptr++;
                        }
                    }
                }

                if (interface_ptr < global_pcount) {
                    interface_list[{cur_patch_idx, interface_ptr}] = InterfaceCom{u64_max,u64_max,u64_max,u64_max,vectype{},vectype{}};
                }
            });
        });

        // // now the list of interface is known
        // {
        //     auto interface_list = interface_list_buf.get_access<sycl::access::mode::read>();

        //     for (u64 i = 0; i < local_pcount; i++) {
        //         std::cout << "- " << sched.patch_list.local[i].id_patch << " : ";
        //         for (u64 j = 0; j < global_pcount; j++) {
        //             if (interface_list[{i, j}].x() == u64_max)
        //                 break;
        //             std::cout << "(" << sched.patch_list.local[interface_list[{i, j}].x()].id_patch << ","
        //                       << sched.patch_list.global[interface_list[{i, j}].y()].id_patch << ") ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        return interface_list_buf;
    }

  public:
    /**
     * @brief
     *
     * @param sched
     * @param sptree
     * @param pfield the interaction radius field
     */
    inline static void gen_interfaces_test(SchedulerMPI &sched, SerialPatchTree<vectype> &sptree,
                                           PatchField<typename vectype::element_type> pfield,std::string fout) {

        SyCLHandler &hndl = SyCLHandler::get_instance();

        const u64 local_pcount  = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        if (local_pcount == 0)
            return;

        sycl::buffer<InterfaceCom, 2> interface_list_buf = get_interface_list_v1(sched, sptree, pfield);

        std::ofstream write_out(fout);

        {
            auto interface_list = interface_list_buf.template get_access<sycl::access::mode::read>();

            for (u64 i = 0; i < local_pcount; i++) {
                std::cout << "- " << sched.patch_list.local[i].id_patch << " : ";
                for (u64 j = 0; j < global_pcount; j++) {
                    if (interface_list[{i, j}].sender_patch_id == u64_max)
                        break;
                    std::cout << "(" << interface_list[{i, j}].sender_patch_id << ","
                              << interface_list[{i, j}].receiver_patch_id << ") ";

                    write_out << 
                        interface_list[{i, j}].local_patch_idx_send << "|" <<
                        interface_list[{i, j}].global_patch_idx_recv << "|" <<
                        interface_list[{i, j}].sender_patch_id << "|" <<
                        interface_list[{i, j}].receiver_patch_id << "|" <<
                        interface_list[{i, j}].interf_box_min.x() << "|" <<
                        interface_list[{i, j}].interf_box_max.x() << "|" << 
                        interface_list[{i, j}].interf_box_min.y() << "|" <<
                        interface_list[{i, j}].interf_box_max.y() << "|" << 
                        interface_list[{i, j}].interf_box_min.z() << "|" <<
                        interface_list[{i, j}].interf_box_max.z() << "|" << "\n";


                }
                std::cout << std::endl;
            }

        }

        write_out.close();

    }
};

/*



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
    inline void gen_interfaces(SchedulerMPI &sched, SerialPatchTree<vectype> &sptree,
                               PatchField<typename vectype::element_type> pfield) {

        SyCLHandler & hndl = SyCLHandler::get_instance();

        const u64 local_pcount = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();

        sycl::buffer<u64> patch_ids_buf(local_pcount);
        sycl::buffer<vectype> local_box_min_buf(local_pcount);
        sycl::buffer<vectype> local_box_max_buf(local_pcount);
        sycl::buffer<u64_2, 2> interface_list_buf({local_pcount,global_pcount});
        sycl::buffer<u64, 2> stack_buf({local_pcount,global_pcount});
        sycl::buffer<u64> stack_start_idx_buf(local_pcount);

        {
            auto pid      = patch_ids_buf.get_access<sycl::access::mode::discard_write>();
            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read_write>();
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read_write>();

            auto stack = stack_buf.get_access<sycl::access::mode::discard_write>();
            auto stack_start_idx = stack_start_idx_buf.get_access<sycl::access::mode::discard_write>();

            std::tuple<vectype,vectype> box_transform = sched.get_box_tranform<vectype>();

            for(u64 i = 0 ; i < local_pcount ; i ++){
                pid[i] = sched.patch_list.local[i].id_patch;

                lbox_min[i] = vectype{sched.patch_list.local[i].x_min, sched.patch_list.local[i].y_min,
   sched.patch_list.local[i].z_min} * std::get<1>(box_transform) + std::get<0>(box_transform); lbox_max[i] =
   (vectype{sched.patch_list.local[i].x_max, sched.patch_list.local[i].y_max, sched.patch_list.local[i].z_max} + 1) *
   std::get<1>(box_transform) + std::get<0>(box_transform);

                lbox_min[i] -= pfield.local_nodes_value[i];
                lbox_max[i] += pfield.local_nodes_value[i];

                //TODO use root_ids list to init the stack instead of just 0
                stack[{i,0}] = 0;
                stack_start_idx[i] = 0;
            }
        }



        PatchFieldReduction<typename vectype::element_type> pfield_reduced =
            sptree.template reduce_field<typename vectype::element_type, OctreeMaxReducer>(hndl.alt_queues[0], sched,
   pfield);



        hndl.alt_queues[0].submit([&](cl::sycl::handler &cgh) {

            auto pid      = patch_ids_buf.get_access<sycl::access::mode::read>();
            auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::read>();
            auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::read>();


            auto interface_list = interface_list_buf.get_access<sycl::access::mode::discard_write>(cgh);

            auto stack = stack_buf.get_access<sycl::access::mode::read_write>();
            auto stack_start_idx = stack_start_idx_buf.get_access<sycl::access::mode::read>();


            cgh.parallel_for(sycl::range<1>(local_pcount), [=](cl::sycl::item<1> item) {

                u64 cur_patch_idx = (u64)item.get_id(0);
                u64 cur_patch_id = pid[cur_patch_idx];
                u64 cur_lbox_min = lbox_min[cur_patch_idx];
                u64 cur_lbox_max = lbox_max[cur_patch_idx];


                u64 interface_ptr = 0;


                u64 current_stack_ptr = stack_start_idx[cur_patch_idx];

                while(current_stack_ptr != u64_max){

                    //pop stack
                    u64 cur_stack_idx = stack[{cur_patch_idx,current_stack_ptr}];
                    if(current_stack_ptr == 0){
                        current_stack_ptr = u64_max;
                    } else {
                        current_stack_ptr --;
                    }




                }

            });

        });

    }

*/