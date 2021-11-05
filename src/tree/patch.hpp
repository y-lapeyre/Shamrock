#pragma once

#include "../aliases.hpp"
#include "../flags.hpp"
#include <cstddef>
#include <vector>

#define add_field_bool(name) inline bool use_field_##name = false
#define add_field_data(name,type) std::vector<type> name


#define add_field_serializer(name,primitive_type,dimension)\
        if(use_field_##name){\
            for(u32 i = 0; i < obj_cnt; i ++){\
                u8* ptr = (u8*) &name[i];\
                buffer.insert(buffer.end(),ptr,ptr + sizeof(primitive_type)*dimension);\
            }\
        }


#define add_field_deserializer_1(name,primitive_type)\
        if(use_field_##name){\
            primitive_type* ptr_f = (primitive_type*) (buffer + offset);\
            for(u32 i = 0; i < obj_cnt; i ++){\
                name.push_back(ptr_f[i]);\
            }\
            offset += sizeof(primitive_type)*obj_cnt;\
        }

#define add_field_deserializer_3(name,primitive_type)\
        if(use_field_##name){\
            primitive_type* ptr_f = (primitive_type*) (buffer + offset);\
            for(u32 i = 0; i < obj_cnt; i ++){\
                name.push_back({ptr_f[i*3],ptr_f[i*3+1],ptr_f[i*3+2]});\
            }\
            offset += 3*sizeof(primitive_type)*obj_cnt;\
        }




//this one is mandatory
const bool use_field_r = true;

add_field_bool(rho);
add_field_bool(v);





class PatchData{
    public:



    u32 obj_cnt = 0;

    add_field_data(r,f3_d);

    add_field_data(rho,f_d);
    add_field_data(v  ,f3_d);




    std::vector<u8> serialize(){
        std::vector<u8> buffer;

        u8* ptr = reinterpret_cast<u8*>(&obj_cnt);

        buffer.insert(buffer.end(),ptr,ptr + sizeof(u32));

        add_field_serializer(r  ,f_d,3)

        add_field_serializer(rho,f_d,1)
        add_field_serializer(v  ,f_d,3)

        return buffer;
    }

    PatchData(std::vector<u8> serialized_data){
        u8* buffer = serialized_data.data();

        obj_cnt = * ((u32*)buffer);

        auto offset = sizeof(u32);

        add_field_deserializer_3(r  ,f_d)

        add_field_deserializer_1(rho,f_d)
        add_field_deserializer_3(v  ,f_d)
    }

    PatchData(){};

};





class Patch{

    u64_3 patch_pos_min;
    u64_3 patch_pos_max;

    u32 patch_level;

    PatchData* data = nullptr;

    u64 patch_id = ERR_ID_64;
    std::vector<u64>* neight_patch_id = nullptr;

    
};



struct Patch_tree_cell{
    u32 patch_tree_level = 0;
    u64_3 patch_pos_min;
    u64_3 patch_pos_max;
    u64 patch_id = ERR_ID_64;
    u64 child_id[8] = {ERR_ID_64,ERR_ID_64,ERR_ID_64,ERR_ID_64,ERR_ID_64,ERR_ID_64,ERR_ID_64,ERR_ID_64};
};

class Patch_Tree{
    f3_d box_min;
    f3_d box_max;

    //63 level max in patch tree
    u64_3 box_size_int = {9223372036854775808u,9223372036854775808u,9223372036854775808u};

    std::vector<Patch_tree_cell> patch_tree_cell_list;

};