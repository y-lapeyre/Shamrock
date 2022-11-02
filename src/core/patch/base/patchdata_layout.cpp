// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "patchdata_layout.hpp"

#define __add_field_desc_gen(T) \
    for (auto f : fields_##T){\
        ss << f.name << " : nvar=" <<f.nvar << " type : " << #T << "\n";\
    }

#define __append_fields_names(vec,T) \
    for (auto f : fields_##T){\
        vec.push_back(f.name);\
    }
    



std::string PatchDataLayout::get_description_str(){
    std::stringstream ss;

    #define X(f) __add_field_desc_gen(f);
    XMAC_LIST_ENABLED_FIELD
    #undef X

    return ss.str();
}

std::vector<std::string> PatchDataLayout::get_field_names(){
    std::vector<std::string> ret ;

    #define X(f) __append_fields_names(ret,f);
    XMAC_LIST_ENABLED_FIELD
    #undef X

    return ret;
}




#undef __add_field_desc_gen
#undef __append_fields_names