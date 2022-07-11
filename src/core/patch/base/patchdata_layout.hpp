// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/patch/base/enabled_fields.hpp"
#include <sstream>
#include <vector>



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

    #define X(f) __add_field_type(f);
    XMAC_LIST_ENABLED_FIELD
    #undef X


    std::string get_description_str();

    std::vector<std::string> get_field_names();

};

#undef __add_field_type
