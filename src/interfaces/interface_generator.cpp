#include "interface_generator.hpp"

#include <stdexcept>
#include <vector>

#include "aliases.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include "utils/geometry_utils.hpp"


template <>
std::vector<PatchData> InterfaceVolumeGenerator::build_interface<f32_3>(sycl::queue &queue, PatchDataBuffer pdat_buf,
                                                                        std::vector<f32_3> boxs_min,
                                                                        std::vector<f32_3> boxs_max) {

    if (boxs_min.size() > u8_max - 1) {
        throw std::runtime_error("this algo is not build to handle more than 2^8 - 2 boxes as input");
    }

    std::vector<u8> flag_choice(pdat_buf.element_count);

    {
        sycl::buffer<u8> flag_buf(flag_choice);

        sycl::buffer<f32_3> bmin_buf(boxs_min);
        sycl::buffer<f32_3> bmax_buf(boxs_max);

        cl::sycl::range<1> range{pdat_buf.element_count};

        queue.submit([&](cl::sycl::handler &cgh) {
            auto pos_s = pdat_buf.pos_s.get_access<sycl::access::mode::read>(cgh);

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

    std::vector<PatchData> pdat_ret(boxs_min.size());

    {

        auto pos_s = pdat_buf.pos_s.get_access<sycl::access::mode::read>();
        auto pos_d = pdat_buf.pos_d.get_access<sycl::access::mode::read>();
        auto U1_s  = pdat_buf.U1_s.get_access<sycl::access::mode::read>();
        auto U1_d  = pdat_buf.U1_d.get_access<sycl::access::mode::read>();
        auto U3_s  = pdat_buf.U3_s.get_access<sycl::access::mode::read>();
        auto U3_d  = pdat_buf.U3_d.get_access<sycl::access::mode::read>();

        for (u32 idx = 0; idx < pdat_buf.element_count; idx++) {
            if (flag_choice[idx] < pdat_ret.size()) {
                PatchData &pdat = pdat_ret[flag_choice[idx]];

                pdat.pos_s.push_back(pos_s[idx]);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U1_s.push_back(U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U1_d.push_back(U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U3_s.push_back(U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat.U3_d.push_back(U3_d[idx * patchdata_layout::nVarU3_d + j]);
                }
            }
        }
    }

    return pdat_ret;
}

template <>
std::vector<PatchData> InterfaceVolumeGenerator::build_interface<f64_3>(sycl::queue &queue, PatchDataBuffer pdat_buf,
                                                                        std::vector<f64_3> boxs_min,
                                                                        std::vector<f64_3> boxs_max) {

    if (boxs_min.size() > u8_max - 1) {
        throw std::runtime_error("this algo is not build to handle more than 2^8 - 2 boxes as input");
    }

    std::vector<u8> flag_choice(pdat_buf.element_count);

    {
        sycl::buffer<u8> flag_buf(flag_choice);

        sycl::buffer<f64_3> bmin_buf(boxs_min);
        sycl::buffer<f64_3> bmax_buf(boxs_max);

        cl::sycl::range<1> range{pdat_buf.element_count};

        queue.submit([&](cl::sycl::handler &cgh) {
            auto pos_d = pdat_buf.pos_d.get_access<sycl::access::mode::read>(cgh);

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

    std::vector<PatchData> pdat_ret(boxs_min.size());

    {

        auto pos_s = pdat_buf.pos_s.get_access<sycl::access::mode::read>();
        auto pos_d = pdat_buf.pos_d.get_access<sycl::access::mode::read>();
        auto U1_s  = pdat_buf.U1_s.get_access<sycl::access::mode::read>();
        auto U1_d  = pdat_buf.U1_d.get_access<sycl::access::mode::read>();
        auto U3_s  = pdat_buf.U3_s.get_access<sycl::access::mode::read>();
        auto U3_d  = pdat_buf.U3_d.get_access<sycl::access::mode::read>();

        for (u32 idx = 0; idx < pdat_buf.element_count; idx++) {
            if (flag_choice[idx] < pdat_ret.size()) {
                PatchData &pdat = pdat_ret[flag_choice[idx]];

                pdat.pos_d.push_back(pos_d[idx]);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U1_s.push_back(U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U1_d.push_back(U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat.U3_s.push_back(U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat.U3_d.push_back(U3_d[idx * patchdata_layout::nVarU3_d + j]);
                }
            }
        }
    }

    return pdat_ret;
}