#pragma once

#include <vector>

#include "aliases.hpp"

template<class T>
class PatchDataField {

    std::vector<T> field_data;

    std::string field_name;

    u32 nvar;
    u32 obj_cnt;

    u32 val_cnt;

    public:

    using Field_type = T;

    inline PatchDataField(std::string name, u32 nvar) : field_name(name) , nvar(nvar){

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

    inline void resize(u32 new_obj_cnt){
        field_data.resize(new_obj_cnt*nvar);
        obj_cnt = new_obj_cnt;
        val_cnt = new_obj_cnt*nvar;
    }

};