#include "interface_generator.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "CL/sycl/buffer.hpp"
#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include "utils/geometry_utils.hpp"


template <>
std::vector<std::unique_ptr<PatchData>> InterfaceVolumeGenerator::append_interface<f32_3>(sycl::queue &queue, PatchData & pdat,
                                                                        std::vector<f32_3> boxs_min,
                                                                        std::vector<f32_3> boxs_max) {

    if (boxs_min.size() > u8_max - 1) {
        throw std::runtime_error("this algo is not build to handle more than 2^8 - 2 boxes as input");
    }

    
        
    


    std::vector<u8> flag_choice(pdat.pos_s.size());

    if (pdat.pos_s.size() > 0) {
    
        sycl::buffer<u8> flag_buf(flag_choice);

        sycl::buffer<f32_3> bmin_buf(boxs_min);
        sycl::buffer<f32_3> bmax_buf(boxs_max);

        cl::sycl::range<1> range{pdat.pos_s.size()};

        sycl::buffer<f32_3> pos_s_buf = sycl::buffer<f32_3>(pdat.pos_s);

        queue.submit([&](cl::sycl::handler &cgh) {
            auto pos_s = pos_s_buf.get_access<sycl::access::mode::read>(cgh);

            auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
            auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

            auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

            u8 num_boxes = boxs_min.size();

            cgh.parallel_for<class BuildInterfacef32>(range, [=](cl::sycl::item<1> item) {
                u64 i = (u64)item.get_id(0);

                f32_3 pos_i  = pos_s[i];
                index_box[i] = u8_max;

                for (u8 idx = 0; idx < num_boxes; idx++) {
                    if (BBAA::is_particle_in_patch<f32_3>(pos_i, bmin[idx], bmax[idx])) {
                        index_box[i] = idx;
                    }
                }
            });
        });
    
    }

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>();
    }

    if (pdat.pos_s.size() > 0) {

        for (u32 idx = 0; idx < pdat.pos_s.size(); idx++) {
            if (flag_choice[idx] < pdat_vec.size()) {

                pdat_vec[flag_choice[idx]]->pos_s.push_back(pdat.pos_s[idx]);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_s.push_back(pdat.U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_d.push_back(pdat.U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_s.push_back(pdat.U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_d.push_back(pdat.U3_d[idx * patchdata_layout::nVarU3_d + j]);
                }
            }
        }
    }

    return pdat_vec;

}

template <>
std::vector<std::unique_ptr<PatchData>> InterfaceVolumeGenerator::append_interface<f64_3>(sycl::queue &queue, PatchData & pdat,
                                                                        std::vector<f64_3> boxs_min,
                                                                        std::vector<f64_3> boxs_max) {

    if (boxs_min.size() > u8_max - 1) {
        throw std::runtime_error("this algo is not build to handle more than 2^8 - 2 boxes as input");
    }

    std::vector<u8> flag_choice(pdat.pos_d.size());

    if (pdat.pos_d.size() > 0) {
        sycl::buffer<u8> flag_buf(flag_choice);

        sycl::buffer<f64_3> bmin_buf(boxs_min);
        sycl::buffer<f64_3> bmax_buf(boxs_max);

        cl::sycl::range<1> range{pdat.pos_d.size()};

        sycl::buffer<f64_3> pos_d_buf = sycl::buffer<f64_3>(pdat.pos_d);

        queue.submit([&](cl::sycl::handler &cgh) {
            auto pos_d = pos_d_buf.get_access<sycl::access::mode::read>(cgh);

            auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
            auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

            auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

            u8 num_boxes = boxs_min.size();

            

            cgh.parallel_for<class BuildInterfacef64>(range, [=](cl::sycl::item<1> item) {
                u64 i = (u64)item.get_id(0);

                f64_3 pos_i  = pos_d[i];
                index_box[i] = u8_max;

                for (u8 idx = 0; idx < num_boxes; idx++) {
                    if (BBAA::is_particle_in_patch(pos_i, bmin[idx], bmax[idx])) {
                        index_box[i] = idx;
                    }
                }
            });
        });
    }
    
    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>();
    }

    if (pdat.pos_d.size() > 0) {


        for (u32 idx = 0; idx < pdat.pos_d.size(); idx++) {
            if (flag_choice[idx] < pdat_vec.size()) {


                pdat_vec[flag_choice[idx]]->pos_d.push_back(pdat.pos_d[idx]);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_s.push_back(pdat.U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_d.push_back(pdat.U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_s.push_back(pdat.U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_d.push_back(pdat.U3_d[idx * patchdata_layout::nVarU3_d + j]);
                }
            }
        }
    }

    return pdat_vec;

}




