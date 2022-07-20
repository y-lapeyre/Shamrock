#include "merged_patch.hpp"
#include "core/patch/base/enabled_fields.hpp"
#include "core/patch/base/patchdata.hpp"

template<class flt>
auto MergedPatchData<flt>::merge_patches(
    PatchScheduler & sched,
    InterfaceHandler<vec, flt> & interface_hndl) -> std::unordered_map<u64,MergedPatchData<flt>> {

    std::unordered_map<u64,MergedPatchData<flt>> merged_data;
    
    sched.for_each_patch_data([&](u64 id_patch, Patch & p, PatchData & pdat){


        merged_data.emplace(id_patch,sched.pdl);

        auto pbox = sched.patch_data.sim_box.get_box<flt>(p);
        u32 original_element = pdat.get_obj_cnt();

        MergedPatchData<flt> & ret = merged_data.at(id_patch);
        
        ret.data.insert_elements(pdat);
        
        interface_hndl.for_each_interface(
            id_patch, 
            [&](u64 patch_id, u64 interf_patch_id, PatchData & interfpdat, std::tuple<vec,vec> box){

                std::get<0>(pbox) = sycl::min(std::get<0>(box),std::get<0>(pbox));
                std::get<1>(pbox) = sycl::min(std::get<1>(box),std::get<1>(pbox));

                ret.data.insert_elements(interfpdat);

            }
        );

        ret.box = pbox;
        ret.or_element_cnt = original_element;


    });

    return merged_data;

}



template<class flt, class T>
auto MergedPatchCompField<flt,T>::merge_patches_cfield(
    PatchScheduler & sched,
    InterfaceHandler<vec, flt> & interface_hndl,
    PatchComputeField<f32> & comp_field,
    PatchComputeFieldInterfaces<f32> & comp_field_interf) -> std::unordered_map<u64,MergedPatchCompField<flt,T>> {


    std::unordered_map<u64,MergedPatchCompField<flt,T>> merged_data;


    sched.for_each_patch([&](u64 id_patch, Patch cur_p) {




    });


}



template class MergedPatchData<f32>;
template class MergedPatchData<f64>;


#define X(arg)\
template class MergedPatchCompField<f32,arg>;\
template class MergedPatchCompField<f64,arg>;
XMAC_LIST_ENABLED_FIELD
#undef X