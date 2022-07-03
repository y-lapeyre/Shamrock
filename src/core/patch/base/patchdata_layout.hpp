// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include <sstream>
#include <vector>



//TODO WIP module

template<class T>
class FieldDescriptor{public: 

    std::string name;
    u32 nvar;
    using field_T = T;

    inline FieldDescriptor(){};

    inline FieldDescriptor(std::string name, u32 nvar){
        this->name = name;
        this->nvar = nvar;
    }
};

#define __add_field_type(T) \
    std::vector<FieldDescriptor< T >> fields_##T;\
    template<> inline void add_field<T>(std::string field_name, u32 nvar){fields_##T.push_back(FieldDescriptor<T>(field_name,nvar));}\
    template<> [[nodiscard]] inline FieldDescriptor<T> get_field<T>(std::string field_name){    \
                                                                                      \
        bool found = false;                                                           \
        FieldDescriptor<T> ret;                                                       \
                                                                                      \
        for (auto a : fields_##T) {                                                     \
            if(a.name == field_name){                                                  \
                if(found) throw shamrock_exc("field ("+field_name+") exist multiple times");      \
                ret = a; found = true;                                                 \
            }                                                                          \
        }                                                                               \
                                                                                        \
                                                                           \
        if(!found) throw shamrock_exc("field ("+field_name+") not found");                                                               \
                                                                                 \
        return ret;                                                               \
                                                                                       \
                                                                                       \
    }                                                               \
    template<> [[nodiscard]] inline u32 get_field_idx<T>(std::string field_name){    \
                                                                                      \
        bool found = false;                                                           \
        u32 ret;                                                       \
                                                                                      \
        for (u32 idx = 0; idx < fields_##T.size() ; idx++) {        \
            auto a = fields_##T[idx];                                             \
            if(a.name == field_name){                                                  \
                if(found) throw shamrock_exc("field ("+field_name+") exist multiple times");      \
                ret = idx; found = true;                                                 \
            }                                                                          \
        }                                                                               \
                                                                                        \
                                                                           \
        if(!found) throw shamrock_exc("field ("+field_name+") not found");                                                               \
                                                                                 \
        return ret;                                                               \
                                                                                       \
                                                                                       \
    }   


#define __add_field_desc_gen(T) \
    for (auto f : fields_##T){\
        ss << f.name << " : nvar=" <<f.nvar << " type : " << #T << "\n";\
    }

#define __append_fields_names(vec,T) \
    for (auto f : fields_##T){\
        vec.push_back(f.name);\
    }
    



enum PositionprecMode{
    xyz32,xyz64
};

class PatchDataLayout {

    //TODO add MPI sync

public:
    template<class T>
    void add_field(std::string field_name, u32 nvar);

    template<class T>
    FieldDescriptor<T> get_field(std::string field_name);

    template<class T>
    u32 get_field_idx(std::string field_name);

    PositionprecMode xyz_mode;


    __add_field_type(i64   );
    __add_field_type(i32   );
    //__add_field_type(i16   );
    //__add_field_type( i8   );
    __add_field_type(u64   );
    __add_field_type(u32   );
    //__add_field_type(u16   );
    //__add_field_type( u8   );
    //__add_field_type(f16   );
    __add_field_type(f32   );
    __add_field_type(f64   );
    //__add_field_type(i64_2 );
    //__add_field_type(i32_2 );
    //__add_field_type(i16_2 );
    //__add_field_type( i8_2 );
    //__add_field_type(u64_2 );
    //__add_field_type(u32_2 );
    //__add_field_type(u16_2 );
    //__add_field_type( u8_2 );
    //__add_field_type(f16_2 );
    __add_field_type(f32_2 );
    __add_field_type(f64_2 );
    //__add_field_type(i64_3 );
    //__add_field_type(i32_3 );
    //__add_field_type(i16_3 );
    //__add_field_type( i8_3 );
    //__add_field_type(u64_3 );
    //__add_field_type(u32_3 );
    //__add_field_type(u16_3 );
    //__add_field_type( u8_3 );
    //__add_field_type(f16_3 );
    __add_field_type(f32_3 );
    __add_field_type(f64_3 );
    //__add_field_type(i64_4 );
    //__add_field_type(i32_4 );
    //__add_field_type(i16_4 );
    //__add_field_type( i8_4 );
    //__add_field_type(u64_4 );
    //__add_field_type(u32_4 );
    //__add_field_type(u16_4 );
    //__add_field_type( u8_4 );
    //__add_field_type(f16_4 );
    __add_field_type(f32_4 );
    __add_field_type(f64_4 );
    //__add_field_type(i64_8 );
    //__add_field_type(i32_8 );
    //__add_field_type(i16_8 );
    //__add_field_type( i8_8 );
    //__add_field_type(u64_8 );
    //__add_field_type(u32_8 );
    //__add_field_type(u16_8 );
    //__add_field_type( u8_8 );
    //__add_field_type(f16_8 );
    __add_field_type(f32_8 );
    __add_field_type(f64_8 );
    //__add_field_type(i64_16);
    //__add_field_type(i32_16);
    //__add_field_type(i16_16);
    //__add_field_type( i8_16);
    //__add_field_type(u64_16);
    //__add_field_type(u32_16);
    //__add_field_type(u16_16);
    //__add_field_type( u8_16);
    //__add_field_type(f16_16);
    __add_field_type(f32_16);
    __add_field_type(f64_16);


    inline std::string get_description_str(){
        std::stringstream ss;

        __add_field_desc_gen(i64   );
        __add_field_desc_gen(i32   );
        //__add_field_desc_gen(i16   );
        //__add_field_desc_gen( i8   );
        __add_field_desc_gen(u64   );
        __add_field_desc_gen(u32   );
        //__add_field_desc_gen(u16   );
        //__add_field_desc_gen( u8   );
        //__add_field_desc_gen(f16   );
        __add_field_desc_gen(f32   );
        __add_field_desc_gen(f64   );
        //__add_field_desc_gen(i64_2 );
        //__add_field_desc_gen(i32_2 );
        //__add_field_desc_gen(i16_2 );
        //__add_field_desc_gen( i8_2 );
        //__add_field_desc_gen(u64_2 );
        //__add_field_desc_gen(u32_2 );
        //__add_field_desc_gen(u16_2 );
        //__add_field_desc_gen( u8_2 );
        //__add_field_desc_gen(f16_2 );
        __add_field_desc_gen(f32_2 );
        __add_field_desc_gen(f64_2 );
        //__add_field_desc_gen(i64_3 );
        //__add_field_desc_gen(i32_3 );
        //__add_field_desc_gen(i16_3 );
        //__add_field_desc_gen( i8_3 );
        //__add_field_desc_gen(u64_3 );
        //__add_field_desc_gen(u32_3 );
        //__add_field_desc_gen(u16_3 );
        //__add_field_desc_gen( u8_3 );
        //__add_field_desc_gen(f16_3 );
        __add_field_desc_gen(f32_3 );
        __add_field_desc_gen(f64_3 );
        //__add_field_desc_gen(i64_4 );
        //__add_field_desc_gen(i32_4 );
        //__add_field_desc_gen(i16_4 );
        //__add_field_desc_gen( i8_4 );
        //__add_field_desc_gen(u64_4 );
        //__add_field_desc_gen(u32_4 );
        //__add_field_desc_gen(u16_4 );
        //__add_field_desc_gen( u8_4 );
        //__add_field_desc_gen(f16_4 );
        __add_field_desc_gen(f32_4 );
        __add_field_desc_gen(f64_4 );
        //__add_field_desc_gen(i64_8 );
        //__add_field_desc_gen(i32_8 );
        //__add_field_desc_gen(i16_8 );
        //__add_field_desc_gen( i8_8 );
        //__add_field_desc_gen(u64_8 );
        //__add_field_desc_gen(u32_8 );
        //__add_field_desc_gen(u16_8 );
        //__add_field_desc_gen( u8_8 );
        //__add_field_desc_gen(f16_8 );
        __add_field_desc_gen(f32_8 );
        __add_field_desc_gen(f64_8 );
        //__add_field_desc_gen(i64_16);
        //__add_field_desc_gen(i32_16);
        //__add_field_desc_gen(i16_16);
        //__add_field_desc_gen( i8_16);
        //__add_field_desc_gen(u64_16);
        //__add_field_desc_gen(u32_16);
        //__add_field_desc_gen(u16_16);
        //__add_field_desc_gen( u8_16);
        //__add_field_desc_gen(f16_16);
        __add_field_desc_gen(f32_16);
        __add_field_desc_gen(f64_16);


        return ss.str();
    }

    inline std::vector<std::string> get_field_names(){
        std::vector<std::string> ret ;

        __append_fields_names(ret,i64   );
        __append_fields_names(ret,i32   );
        //__append_fields_names(ret,i16   );
        //__append_fields_names(ret, i8   );
        __append_fields_names(ret,u64   );
        __append_fields_names(ret,u32   );
        //__append_fields_names(ret,u16   );
        //__append_fields_names(ret, u8   );
        //__append_fields_names(ret,f16   );
        __append_fields_names(ret,f32   );
        __append_fields_names(ret,f64   );
        //__append_fields_names(ret,i64_2 );
        //__append_fields_names(ret,i32_2 );
        //__append_fields_names(ret,i16_2 );
        //__append_fields_names(ret, i8_2 );
        //__append_fields_names(ret,u64_2 );
        //__append_fields_names(ret,u32_2 );
        //__append_fields_names(ret,u16_2 );
        //__append_fields_names(ret, u8_2 );
        //__append_fields_names(ret,f16_2 );
        __append_fields_names(ret,f32_2 );
        __append_fields_names(ret,f64_2 );
        //__append_fields_names(ret,i64_3 );
        //__append_fields_names(ret,i32_3 );
        //__append_fields_names(ret,i16_3 );
        //__append_fields_names(ret, i8_3 );
        //__append_fields_names(ret,u64_3 );
        //__append_fields_names(ret,u32_3 );
        //__append_fields_names(ret,u16_3 );
        //__append_fields_names(ret, u8_3 );
        //__append_fields_names(ret,f16_3 );
        __append_fields_names(ret,f32_3 );
        __append_fields_names(ret,f64_3 );
        //__append_fields_names(ret,i64_4 );
        //__append_fields_names(ret,i32_4 );
        //__append_fields_names(ret,i16_4 );
        //__append_fields_names(ret, i8_4 );
        //__append_fields_names(ret,u64_4 );
        //__append_fields_names(ret,u32_4 );
        //__append_fields_names(ret,u16_4 );
        //__append_fields_names(ret, u8_4 );
        //__append_fields_names(ret,f16_4 );
        __append_fields_names(ret,f32_4 );
        __append_fields_names(ret,f64_4 );
        //__append_fields_names(ret,i64_8 );
        //__append_fields_names(ret,i32_8 );
        //__append_fields_names(ret,i16_8 );
        //__append_fields_names(ret, i8_8 );
        //__append_fields_names(ret,u64_8 );
        //__append_fields_names(ret,u32_8 );
        //__append_fields_names(ret,u16_8 );
        //__append_fields_names(ret, u8_8 );
        //__append_fields_names(ret,f16_8 );
        __append_fields_names(ret,f32_8 );
        __append_fields_names(ret,f64_8 );
        //__append_fields_names(ret,i64_16);
        //__append_fields_names(ret,i32_16);
        //__append_fields_names(ret,i16_16);
        //__append_fields_names(ret, i8_16);
        //__append_fields_names(ret,u64_16);
        //__append_fields_names(ret,u32_16);
        //__append_fields_names(ret,u16_16);
        //__append_fields_names(ret, u8_16);
        //__append_fields_names(ret,f16_16);
        __append_fields_names(ret,f32_16);
        __append_fields_names(ret,f64_16);


        return ret;
    }

};
