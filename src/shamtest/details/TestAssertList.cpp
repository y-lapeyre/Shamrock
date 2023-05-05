// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestAssertList.hpp"


namespace shamtest::details {
    
    
    std::string TestAssertList::serialize(){
        std::string acc = "\n[\n";

        for(u32 i = 0; i < asserts.size(); i++){
            acc += shambase::increase_indent( asserts[i].serialize()) ;
            if(i < asserts.size()-1){
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }

}