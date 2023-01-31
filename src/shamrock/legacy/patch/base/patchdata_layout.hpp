// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamsys/legacy/log.hpp"
#include <sstream>
#include <variant>
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







enum PositionprecMode{
    xyz32,xyz64
};

class PatchDataLayout {


    using var_t = std::variant<
        FieldDescriptor<f32   >, 
        FieldDescriptor<f32_2 >, 
        FieldDescriptor<f32_3 >, 
        FieldDescriptor<f32_4 >, 
        FieldDescriptor<f32_8 >, 
        FieldDescriptor<f32_16>, 
        FieldDescriptor<f64   >, 
        FieldDescriptor<f64_2 >, 
        FieldDescriptor<f64_3 >, 
        FieldDescriptor<f64_4 >, 
        FieldDescriptor<f64_8 >, 
        FieldDescriptor<f64_16>, 
        FieldDescriptor<u32   >, 
        FieldDescriptor<u64   >, 
        FieldDescriptor<u32_3 >, 
        FieldDescriptor<u64_3 >
        >;

    std::vector<var_t> fields;

public:




    template<class T>
    void add_field(std::string field_name, u32 nvar){
        bool found = false;

        for (var_t & fvar : fields) {
            std::visit([&](auto & arg){
                if(field_name == arg.name){
                    found = true;
                }
            }, fvar);
        }

        if(found){
            throw std::invalid_argument("add_field -> the name already exists");
        }

        logger::info_ln("PatchDataLayout", "adding field :",field_name,nvar);

        fields.push_back(FieldDescriptor<T>(field_name,nvar));
    }

    template<class T>
    inline FieldDescriptor<T> get_field(std::string field_name){

        for (var_t & fvar : fields) {
            if(FieldDescriptor<T>* pval = std::get_if<FieldDescriptor<T>>(&fvar)){
                if(pval->name == field_name){
                    return *pval;
                }
            }
        }

        throw std::invalid_argument("the requested field does not exists\n    current table : " + get_description_str());
    }

    template<class T>
    inline u32 get_field_idx(std::string field_name){
        for (u32 i = 0; i < fields.size(); i++) {
            if(FieldDescriptor<T>* pval = std::get_if<FieldDescriptor<T>>(&fields[i])){
                if(pval->name == field_name){
                    return i;
                }
            }
        }

        throw std::invalid_argument("the requested field does not exists\n    current table : " + get_description_str());
    }

    PositionprecMode xyz_mode;



    std::string get_description_str();

    std::vector<std::string> get_field_names();

    template<class Functor>
    inline void for_each_field_any(Functor && func){
        for(auto & f : fields){
            std::visit([&](auto & arg){
                func(arg);
            },f);
        }
    }

};
