// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestDataList.hpp"
#include "shambase/string.hpp"

namespace shamtest::details {
    
    std::string TestDataList::serialize(){
        std::string acc = "\n[\n";

        for(u32 i = 0; i < test_data.size(); i++){
            acc += shambase::increase_indent( test_data[i].serialize()) ;
            if(i < test_data.size()-1){
                acc += ",";
            }
        }

        acc += "\n]";
        return acc;
    }

}