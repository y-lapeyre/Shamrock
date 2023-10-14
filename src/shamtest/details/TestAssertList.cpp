// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "TestAssertList.hpp"
#include <sstream>


namespace shamtest::details {
    
    
    std::string TestAssertList::serialize_json(){
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


    std::basic_string<u8> TestAssertList::serialize() {

        std::basic_stringstream<u8> out;


        u64 asserts_len = asserts.size();
        out.write(reinterpret_cast<u8 const*>(&asserts_len), sizeof(asserts_len));

        return out.str();
    }

}