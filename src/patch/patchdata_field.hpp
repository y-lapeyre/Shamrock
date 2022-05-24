// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include <vector>

#include "aliases.hpp"

#include "utils/sycl_vector_utils.hpp"

template<class T>
class PatchDataField {

    std::vector<T> field_data;

    std::string field_name;

    u32 nvar;
    u32 obj_cnt;

    u32 val_cnt;

    public:

    //TODO find a way to add particles easily cf setup require public vector
    

    using Field_type = T;

    inline PatchDataField(std::string name, u32 nvar) : field_name(name) , nvar(nvar){
        obj_cnt = 0;
        val_cnt = 0;
    };

    inline T* data(){
        return field_data.data();
    }

    inline u32 size(){
        return val_cnt;
    }

    inline u32 get_nvar(){
        return nvar;
    }

    inline u32 get_obj_cnt(){
        return obj_cnt;
    }

    inline std::string get_name(){
        return field_name;
    }

    inline void resize(u32 new_obj_cnt){
        field_data.resize(new_obj_cnt*nvar);
        obj_cnt = new_obj_cnt;
        val_cnt = new_obj_cnt*nvar;
    }

    inline void expand(u32 obj_to_add){
        resize(obj_cnt + obj_to_add);
    }



    inline void insert_element(T v){
        u32 ins_pos = val_cnt;
        expand(1);
        field_data[ins_pos] = v;
    }

    inline void apply_offset(T off){
        for(T & v : field_data){
            v += off;
        }
    }


    inline void insert(PatchDataField<T> &f2){
        field_data.insert(field_data.end(), f2.field_data.begin(), f2.field_data.end());
    }

    inline bool check_field_match(PatchDataField<T> &f2){
        bool match = true;

        match = match && (field_name == f2.field_name);
        match = match && (nvar       == f2.nvar);
        match = match && (obj_cnt    == f2.obj_cnt);
        match = match && (val_cnt    == f2.val_cnt);

        for (u32 i = 0; i < val_cnt; i++) {
            match = match && test_sycl_eq(data()[i],f2.data()[i]);
        }

        return match;
    }

    /**
     * @brief Copy all objects in idxs to pfield
     * 
     * @param idxs 
     * @param pfield 
     */
    inline void append_subset_to(std::vector<u32> & idxs, PatchDataField & pfield){

        if(pfield.nvar != nvar) throw shamrock_exc("field must be similar for extraction");

        const u32 start_enque = pfield.val_cnt;

        const u32 nvar = get_nvar();

        pfield.expand(idxs.size());

        for (u32 i = 0; i < idxs.size(); i++) {

            const u32 idx_extr = idxs[i]*nvar;
            const u32 idx_push = start_enque + i*nvar;

            for(u32 a = 0; a < nvar ; a++){
                pfield.data()[idx_push + a] = data()[idx_extr + a];
            }

        }
    }

};

