// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "PatchData.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/legacy/utils/geometry_utils.hpp"

#include "Patch.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

namespace shamrock::patch{



    PatchData PatchData::mock_patchdata(u64 seed, u32 obj_cnt, PatchDataLayout &pdl){
        PatchData pdat{pdl};

        pdat.fields.clear();

        pdl.for_each_field_any([&](auto & field){
            using f_t = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            pdat.fields.push_back(var_t{
                PatchDataField<base_t>::mock_field(seed,obj_cnt,field.name,field.nvar)
                });

        });

        return pdat;
    }
    


    void PatchData::init_fields(){

        pdl.for_each_field_any([&](auto & field){
            using f_t = typename std::remove_reference<decltype(field)>::type;
            using base_t = typename f_t::field_T;

            fields.push_back(var_t{PatchDataField<base_t>(field.name,field.nvar)});

        });

    }





    void PatchData::extract_element(u32 pidx, PatchData & out_pdat){
        StackEntry stack_loc{};

        for(u32 idx = 0; idx < fields.size(); idx++){

            std::visit([&](auto & field, auto & out_field) {

                using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    field.extract_element(pidx,out_field);
                }else{  
                    throw shambase::throw_with_loc<std::invalid_argument>("missmatch");
                }

            }, fields[idx].value, out_pdat.fields[idx].value);

        }

    }

    void PatchData::insert_elements(PatchData & pdat){
        
        StackEntry stack_loc{};

        for(u32 idx = 0; idx < fields.size(); idx++){

            std::visit([&](auto & field, auto & out_field) {

                using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    field.insert(out_field);
                }else{  
                    throw shambase::throw_with_loc<std::invalid_argument>("missmatch");
                }

            }, fields[idx].value, pdat.fields[idx].value);

        }

    }

    void PatchData::overwrite(PatchData &pdat, u32 obj_cnt){
        StackEntry stack_loc{};
        
        for(u32 idx = 0; idx < fields.size(); idx++){

            std::visit([&](auto & field, auto & out_field) {

                using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    field.overwrite(out_field,obj_cnt);
                }else{  
                    throw shambase::throw_with_loc<std::invalid_argument>("missmatch");
                }

            }, fields[idx].value, pdat.fields[idx].value);

        }
    }



    void PatchData::resize(u32 new_obj_cnt){

        for(auto & field_var : fields){
            field_var.visit([&](auto & field){
                field.resize(new_obj_cnt);
            });
        }

    }

    void PatchData::expand(u32 new_obj_cnt){

        for(auto & field_var : fields){
            field_var.visit([&](auto & field){
                field.expand(new_obj_cnt);
            });
        }

    }


    void PatchData::index_remap(sycl::buffer<u32> index_map, u32 len){

        for(auto & field_var : fields){
            field_var.visit([&](auto & field){
                field.index_remap(index_map, len);
            });
        }

    }

    void PatchData::index_remap_resize(sycl::buffer<u32> index_map, u32 len){

        for(auto & field_var : fields){
            field_var.visit([&](auto & field){
                field.index_remap_resize(index_map, len);
            });
        }

    }



    void PatchData::append_subset_to(sycl::buffer<u32> & idxs, u32 sz, PatchData & pdat) const {
        StackEntry stack_loc{};

        for(u32 idx = 0; idx < fields.size(); idx++){

            std::visit([&](auto & field, auto & out_field) {

                using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    field.append_subset_to(idxs, sz, out_field);
                }else{  
                    throw shambase::throw_with_loc<std::invalid_argument>("missmatch");
                }

            }, fields[idx].value, pdat.fields[idx].value);

        }
    }

    void PatchData::append_subset_to(std::vector<u32> & idxs, PatchData &pdat) const {
        StackEntry stack_loc{};

        for(u32 idx = 0; idx < fields.size(); idx++){

            std::visit([&](auto & field, auto & out_field) {

                using t1 = typename std::remove_reference<decltype(field)>::type::Field_type;
                using t2 = typename std::remove_reference<decltype(out_field)>::type::Field_type;

                if constexpr (std::is_same<t1, t2>::value){
                    field.append_subset_to(idxs, out_field);
                }else{  
                    throw shambase::throw_with_loc<std::invalid_argument>("missmatch");
                }

            }, fields[idx].value, pdat.fields[idx].value);

        }

    }

    void PatchData::serialize_buf(shamalgs::SerializeHelper & serializer){
        StackEntry stack_loc{};
        for_each_field_any([&](auto & f){
            f.serialize_buf(serializer);
        });        
    }

    u64 PatchData::serialize_buf_byte_size(){
        u64 sum = 0;
        for_each_field_any([&](auto & f){
            sum += f.serialize_buf_byte_size();
        }); 
        return sum;
    } 

    PatchData PatchData::deserialize_buf(shamalgs::SerializeHelper & serializer, PatchDataLayout & pdl){
        StackEntry stack_loc{};

        return PatchData{
            pdl, [&](auto &pdat_fields) {
                
                pdl.for_each_field_any([&](auto & field){
                    using f_t = typename std::remove_reference<decltype(field)>::type;
                    using base_t = typename f_t::field_T;

                    pdat_fields.push_back(var_t{
                        PatchDataField<base_t>::deserialize_buf(serializer, field.name,field.nvar)
                    });

                });

            }
        };
    }

    template<class T>
    void PatchData::split_patchdata(std::array<std::reference_wrapper<PatchData>,8> pdats, std::array<T, 8> min_box,  std::array<T, 8> max_box){

        StackEntry stack_loc{};

        PatchDataField<T> & main_field = fields[0].get_if_ref_throw<T>();


        auto get_vec_idx = [&](T vmin, T vmax) -> std::vector<u32> {
            return main_field.get_elements_with_range(
                [&](T val,T vmin, T vmax){
                    return Patch::is_in_patch_converted(val, vmin,vmax);
                },
                vmin,vmax
            );
        };

        std::vector<u32> idx_p0 = get_vec_idx(min_box[0],max_box[0]);
        std::vector<u32> idx_p1 = get_vec_idx(min_box[1],max_box[1]);
        std::vector<u32> idx_p2 = get_vec_idx(min_box[2],max_box[2]);
        std::vector<u32> idx_p3 = get_vec_idx(min_box[3],max_box[3]);
        std::vector<u32> idx_p4 = get_vec_idx(min_box[4],max_box[4]);
        std::vector<u32> idx_p5 = get_vec_idx(min_box[5],max_box[5]);
        std::vector<u32> idx_p6 = get_vec_idx(min_box[6],max_box[6]);
        std::vector<u32> idx_p7 = get_vec_idx(min_box[7],max_box[7]);

        u32 el_cnt_new = idx_p0.size()+
                        idx_p1.size()+
                        idx_p2.size()+
                        idx_p3.size()+
                        idx_p4.size()+
                        idx_p5.size()+
                        idx_p6.size()+
                        idx_p7.size();

        if(get_obj_cnt() != el_cnt_new || true){

            using namespace shambase::sycl_utils;

            logger::err_ln("PatchData", "error in patchdata split, the new element count doesn't match the old one");
            
            logger::err_ln("PatchData", min_box[0],max_box[0]);
            logger::err_ln("PatchData", min_box[1],max_box[1]);
            logger::err_ln("PatchData", min_box[2],max_box[2]);
            logger::err_ln("PatchData", min_box[3],max_box[3]);
            logger::err_ln("PatchData", min_box[4],max_box[4]);
            logger::err_ln("PatchData", min_box[5],max_box[5]);
            logger::err_ln("PatchData", min_box[6],max_box[6]);
            logger::err_ln("PatchData", min_box[7],max_box[7]);

            T vmin = g_sycl_min(min_box[0],min_box[1]);
            vmin = g_sycl_min(vmin,min_box[2]);
            vmin = g_sycl_min(vmin,min_box[3]);
            vmin = g_sycl_min(vmin,min_box[4]);
            vmin = g_sycl_min(vmin,min_box[5]);
            vmin = g_sycl_min(vmin,min_box[6]);
            vmin = g_sycl_min(vmin,min_box[7]);

            T vmax = g_sycl_max(max_box[0],max_box[1]);
            vmax = g_sycl_max(vmax,max_box[2]);
            vmax = g_sycl_max(vmax,max_box[3]);
            vmax = g_sycl_max(vmax,max_box[4]);
            vmax = g_sycl_max(vmax,max_box[5]);
            vmax = g_sycl_max(vmax,max_box[6]);
            vmax = g_sycl_max(vmax,max_box[7]);

            main_field.check_err_range(
                [&](T val,T vmin, T vmax){
                    return Patch::is_in_patch_converted(val, vmin,vmax);
                },
                vmin,vmax);

        }

        //TODO create a extract subpatch function

        append_subset_to(idx_p0, pdats[0].get());
        append_subset_to(idx_p1, pdats[1].get());
        append_subset_to(idx_p2, pdats[2].get());
        append_subset_to(idx_p3, pdats[3].get());
        append_subset_to(idx_p4, pdats[4].get());
        append_subset_to(idx_p5, pdats[5].get());
        append_subset_to(idx_p6, pdats[6].get());
        append_subset_to(idx_p7, pdats[7].get());

    }

    template void PatchData::split_patchdata(std::array<std::reference_wrapper<PatchData>,8> pdats, std::array<f32_3, 8> min_box,  std::array<f32_3, 8> max_box);
    template void PatchData::split_patchdata(std::array<std::reference_wrapper<PatchData>,8> pdats, std::array<f64_3, 8> min_box,  std::array<f64_3, 8> max_box);
    template void PatchData::split_patchdata(std::array<std::reference_wrapper<PatchData>,8> pdats, std::array<u32_3, 8> min_box,  std::array<u32_3, 8> max_box);
    template void PatchData::split_patchdata(std::array<std::reference_wrapper<PatchData>,8> pdats, std::array<u64_3, 8> min_box,  std::array<u64_3, 8> max_box);

    #ifdef false
    template<>
    void PatchData::split_patchdata<f32_3>(
        PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
        f32_3 bmin_p0, f32_3 bmin_p1, f32_3 bmin_p2, f32_3 bmin_p3, f32_3 bmin_p4, f32_3 bmin_p5, f32_3 bmin_p6, f32_3 bmin_p7, 
        f32_3 bmax_p0, f32_3 bmax_p1, f32_3 bmax_p2, f32_3 bmax_p3, f32_3 bmax_p4, f32_3 bmax_p5, f32_3 bmax_p6, f32_3 bmax_p7){

        split_patchdata<f32_3>(
            {pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7},
            {bmin_p0, bmin_p1, bmin_p2, bmin_p3, bmin_p4, bmin_p5, bmin_p6, bmin_p7},
            {bmax_p0, bmax_p1, bmax_p2, bmax_p3, bmax_p4, bmax_p5, bmax_p6, bmax_p7});

    }


    template<>
    void PatchData::split_patchdata<f64_3>(
        PatchData &pd0, PatchData &pd1, PatchData &pd2, PatchData &pd3, PatchData &pd4, PatchData &pd5, PatchData &pd6, PatchData &pd7, 
        f64_3 bmin_p0, f64_3 bmin_p1, f64_3 bmin_p2, f64_3 bmin_p3, f64_3 bmin_p4, f64_3 bmin_p5, f64_3 bmin_p6, f64_3 bmin_p7, 
        f64_3 bmax_p0, f64_3 bmax_p1, f64_3 bmax_p2, f64_3 bmax_p3, f64_3 bmax_p4, f64_3 bmax_p5, f64_3 bmax_p6, f64_3 bmax_p7){

        PatchDataField<f64_3 >* pval = std::get_if<PatchDataField<f64_3 >>(&fields[0]);

        if(!pval){
            throw std::invalid_argument("the main field should be at id 0");
        }

        PatchDataField<f64_3> & xyz = * pval;

        auto get_vec_idx = [&](f64_3 vmin, f64_3 vmax) -> std::vector<u32> {
            return xyz.get_elements_with_range(
                [&](f64_3 val,f64_3 vmin, f64_3 vmax){
                    return Patch::is_in_patch_converted(val, vmin,vmax);
                },
                vmin,vmax
            );
        };

        std::vector<u32> idx_p0 = get_vec_idx(bmin_p0,bmax_p0);
        std::vector<u32> idx_p1 = get_vec_idx(bmin_p1,bmax_p1);
        std::vector<u32> idx_p2 = get_vec_idx(bmin_p2,bmax_p2);
        std::vector<u32> idx_p3 = get_vec_idx(bmin_p3,bmax_p3);
        std::vector<u32> idx_p4 = get_vec_idx(bmin_p4,bmax_p4);
        std::vector<u32> idx_p5 = get_vec_idx(bmin_p5,bmax_p5);
        std::vector<u32> idx_p6 = get_vec_idx(bmin_p6,bmax_p6);
        std::vector<u32> idx_p7 = get_vec_idx(bmin_p7,bmax_p7);

        //TODO create a extract subpatch function

        append_subset_to(idx_p0, pd0);
        append_subset_to(idx_p1, pd1);
        append_subset_to(idx_p2, pd2);
        append_subset_to(idx_p3, pd3);
        append_subset_to(idx_p4, pd4);
        append_subset_to(idx_p5, pd5);
        append_subset_to(idx_p6, pd6);
        append_subset_to(idx_p7, pd7);

    }
    #endif

}