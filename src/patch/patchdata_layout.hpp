#pragma once

#include "aliases.hpp"



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
    template<> inline FieldDescriptor<T> get_field<T>(std::string field_name){    \
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
    template<> inline u32 get_field_idx<T>(std::string field_name){    \
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

};
