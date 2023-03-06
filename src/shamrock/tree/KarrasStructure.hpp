// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once



#include "aliases.hpp"


namespace shamrock::tree {


    class TreeStructure{

        public:

        std::unique_ptr<sycl::buffer<u32>>      buf_lchild_id;   // size = internal
        std::unique_ptr<sycl::buffer<u32>>      buf_rchild_id;   // size = internal
        std::unique_ptr<sycl::buffer<u8>>       buf_lchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u8>>       buf_rchild_flag; // size = internal
        std::unique_ptr<sycl::buffer<u32>>      buf_endrange;    // size = internal

        bool is_built(){
            return bool(buf_lchild_id) && bool(buf_rchild_id) && bool(buf_lchild_flag) && bool(buf_rchild_flag) && bool(buf_endrange);
        }

        
    };

}