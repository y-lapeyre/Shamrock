#include "interface_generator.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "aliases.hpp"
#include "patch/patch.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_buffer.hpp"
#include "patchscheduler/scheduler_patch_data.hpp"
#include "utils/geometry_utils.hpp"

#include "interface_generator_impl.hpp"




template <>
std::vector<std::unique_ptr<PatchData>> InterfaceVolumeGenerator::append_interface<f32_3>(sycl::queue &queue, PatchData & pdat,
                                                                        std::vector<f32_3> boxs_min,
                                                                        std::vector<f32_3> boxs_max,f32_3 add_offset) {


    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);

    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>();
        // p->pos_s.reserve(patchdata_layout::nVarpos_d*pdat.pos_s.size()/8);
        // p->U1_s.reserve(patchdata_layout::nVarU1_s*pdat.pos_s.size()/8);
        // p->U1_d.reserve(patchdata_layout::nVarU1_d*pdat.pos_s.size()/8);
        // p->U3_s.reserve(patchdata_layout::nVarU3_s*pdat.pos_s.size()/8);
        // p->U3_d.reserve(patchdata_layout::nVarU3_d*pdat.pos_s.size()/8);
    }

    if (pdat.pos_s.size() > 0) {

        for (u32 idx = 0; idx < pdat.pos_s.size(); idx++) {
            if (flag_choice[idx] < pdat_vec.size()) {

                pdat_vec[flag_choice[idx]]->pos_s.push_back(pdat.pos_s[idx] + add_offset);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_s.push_back(pdat.U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_d; j++) {
                    pdat_vec[flag_choice[idx]]->U1_d.push_back(pdat.U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_s.push_back(pdat.U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_d; j++) {
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
                                                                        std::vector<f64_3> boxs_max,f64_3 add_offset) {

    std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);
    
    std::vector<std::unique_ptr<PatchData>> pdat_vec(boxs_min.size());
    for (auto & p : pdat_vec) {
        p = std::make_unique<PatchData>();
        // p->pos_d.reserve(patchdata_layout::nVarpos_d*pdat.pos_d.size()/8);
        // p->U1_s.reserve(patchdata_layout::nVarU1_s*pdat.pos_d.size()/8);
        // p->U1_d.reserve(patchdata_layout::nVarU1_d*pdat.pos_d.size()/8);
        // p->U3_s.reserve(patchdata_layout::nVarU3_s*pdat.pos_d.size()/8);
        // p->U3_d.reserve(patchdata_layout::nVarU3_d*pdat.pos_d.size()/8);
    }

    if (pdat.pos_d.size() > 0) {


        for (u32 idx = 0; idx < pdat.pos_d.size(); idx++) {
            if (flag_choice[idx] < pdat_vec.size()) {


                pdat_vec[flag_choice[idx]]->pos_d.push_back(pdat.pos_d[idx] + add_offset);
                for (u32 j = 0; j < patchdata_layout::nVarU1_s; j++) {
                    pdat_vec[flag_choice[idx]]->U1_s.push_back(pdat.U1_s[idx * patchdata_layout::nVarU1_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU1_d; j++) {
                    pdat_vec[flag_choice[idx]]->U1_d.push_back(pdat.U1_d[idx * patchdata_layout::nVarU1_d + j]);
                }

                for (u32 j = 0; j < patchdata_layout::nVarU3_s; j++) {
                    pdat_vec[flag_choice[idx]]->U3_s.push_back(pdat.U3_s[idx * patchdata_layout::nVarU3_s + j]);
                }
                for (u32 j = 0; j < patchdata_layout::nVarU3_d; j++) {
                    pdat_vec[flag_choice[idx]]->U3_d.push_back(pdat.U3_d[idx * patchdata_layout::nVarU3_d + j]);
                }
            }
        }
    }

    return pdat_vec;

}




