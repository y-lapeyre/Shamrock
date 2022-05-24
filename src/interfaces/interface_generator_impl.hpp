#pragma once

#include "CL/sycl/stream.hpp"
#include "patch/patchdata.hpp"
#include "patch/patchdata_field.hpp"
#include "utils/geometry_utils.hpp"
#include <vector>


namespace impl {



    // TODO make box list reference
    template<class vectype>
    std::vector<u8> get_flag_choice(sycl::queue &queue, PatchData & pdat,
                                                                            std::vector<vectype> boxs_min,
                                                                            std::vector<vectype> boxs_max);

    template<>
    inline std::vector<u8> get_flag_choice<f32_3>(sycl::queue &queue, PatchData & pdat,
                                                                            std::vector<f32_3> boxs_min,
                                                                            std::vector<f32_3> boxs_max){

        if (boxs_min.size() > u8_max - 1) {
            throw shamrock_exc("this algo is not build to handle more than 2^8 - 2 boxes as input");
        }


        //TODO change this func when implementing the USM patch without the pos_s_buf/pos_d_buf
        std::vector<u8> flag_choice(pdat.get_obj_cnt());

        if (! pdat.is_empty()) {
        
            sycl::buffer<u8> flag_buf(flag_choice.data(),flag_choice.size());

            sycl::buffer<f32_3> bmin_buf(boxs_min.data(),boxs_min.size());
            sycl::buffer<f32_3> bmax_buf(boxs_max.data(),boxs_max.size());

            sycl::range<1> range{pdat.get_obj_cnt()};

            u32 field_ipos = pdat.patchdata_layout.get_field_idx<f32_3>("xyz");

            PatchDataField<f32_3> & pos_field = pdat.fields_f32_3[field_ipos];

            sycl::buffer<f32_3> pos_s_buf = sycl::buffer<f32_3>(pos_field.data(),pos_field.size());

            ;

            queue.submit([&](sycl::handler &cgh) {
                
                auto pos_s = pos_s_buf.get_access<sycl::access::mode::read>(cgh);

                auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
                auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

                auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

                u8 num_boxes = boxs_min.size();


                cgh.parallel_for<class BuildInterfacef32>(range, [=](sycl::item<1> item) {
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

        return flag_choice;
    }

    template<>
    inline std::vector<u8> get_flag_choice<f64_3>(sycl::queue &queue, PatchData & pdat,
                                                                            std::vector<f64_3> boxs_min,
                                                                            std::vector<f64_3> boxs_max){

        if (boxs_min.size() > u8_max - 1) {
            throw shamrock_exc("this algo is not build to handle more than 2^8 - 2 boxes as input");
        }

        std::vector<u8> flag_choice(pdat.get_obj_cnt());

        if (! pdat.is_empty()) {
            sycl::buffer<u8> flag_buf(flag_choice.data(),flag_choice.size());

            sycl::buffer<f64_3> bmin_buf(boxs_min.data(),boxs_min.size());
            sycl::buffer<f64_3> bmax_buf(boxs_max.data(),boxs_max.size());

            sycl::range<1> range{pdat.get_obj_cnt()};

            u32 field_ipos = pdat.patchdata_layout.get_field_idx<f64_3>("xyz");

            PatchDataField<f64_3> & pos_field = pdat.fields_f64_3[field_ipos];

            sycl::buffer<f64_3> pos_d_buf = sycl::buffer<f64_3>(pos_field.data(),pos_field.size());

            queue.submit([&](sycl::handler &cgh) {
                auto pos_d = pos_d_buf.get_access<sycl::access::mode::read>(cgh);

                auto bmin = bmin_buf.get_access<sycl::access::mode::read>(cgh);
                auto bmax = bmax_buf.get_access<sycl::access::mode::read>(cgh);

                auto index_box = flag_buf.get_access<sycl::access::mode::discard_write>(cgh);

                u8 num_boxes = boxs_min.size();

                

                cgh.parallel_for<class BuildInterfacef64>(range, [=](sycl::item<1> item) {
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

        return flag_choice;
    }








    template <class T, class vectype>
    inline std::vector<std::unique_ptr<std::vector<T>>> append_interface_field(sycl::queue &queue, PatchData & pdat, std::vector<T> & pdat_cfield,
                                                                            std::vector<vectype> boxs_min,
                                                                            std::vector<vectype> boxs_max) {

        std::vector<u8> flag_choice = impl::get_flag_choice(queue, pdat, boxs_min, boxs_max);
        
        std::vector<std::unique_ptr<std::vector<T>>> pdat_vec(boxs_min.size());
        for (auto & p : pdat_vec) {
            p = std::make_unique<std::vector<T>>();
        }

        if (pdat_cfield.size() > 0) {
            for (u32 idx = 0; idx < pdat_cfield.size(); idx++) {
                if (flag_choice[idx] < pdat_vec.size()) {

                    pdat_vec[flag_choice[idx]]->push_back(pdat_cfield[idx]);
                }
            }
        }

        return pdat_vec;

    }


} // namespace impl