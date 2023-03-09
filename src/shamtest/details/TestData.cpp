// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestData.hpp"
#include "shamrock/legacy/utils/string_utils.hpp"

namespace shamtest::details {
    
    std::string TestData::serialize(){
        std::string acc = "\n{\n";

        acc += R"(    "dataset_name" : ")" + dataset_name + "\",\n" ;
        acc += R"(    "dataset" : )" "\n    [\n" ;

        for(u32 i = 0; i < dataset.size(); i++){
            acc += increase_indent( dataset[i].serialize()) ;
            if(i < dataset.size()-1){
                acc += ",";
            }
        }

        acc += "]" ;

        acc += "\n}";
        return acc;
    }

}