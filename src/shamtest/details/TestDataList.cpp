// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestDataList.hpp"
#include "shambase/string.hpp"
#include <sstream>

namespace shamtest::details {
    
    std::string TestDataList::serialize_json(){
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

    std::basic_string<u8> TestDataList::serialize() {

        std::basic_stringstream<u8> out;

        u64 test_data_len = test_data.size();
        out.write(reinterpret_cast<u8 const*>(&test_data_len), sizeof(test_data_len));

        return out.str();
    }

}