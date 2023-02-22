// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "random.hpp"
#include "shamalgs/memory/memory.hpp"



namespace shamalgs::random {

    template<>
    u8 mock_value(std::mt19937 & eng, u8 min_bound, u8 max_bound){
        std::uniform_int_distribution<u8> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    u16 mock_value(std::mt19937 & eng, u16 min_bound, u16 max_bound){
        std::uniform_int_distribution<u16> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    u32 mock_value(std::mt19937 & eng, u32 min_bound, u32 max_bound){
        std::uniform_int_distribution<u32> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    u64 mock_value(std::mt19937 & eng, u64 min_bound, u64 max_bound){
        std::uniform_int_distribution<u64> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    i8 mock_value(std::mt19937 & eng, i8 min_bound, i8 max_bound){
        std::uniform_int_distribution<i8> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    i16 mock_value(std::mt19937 & eng, i16 min_bound, i16 max_bound){
        std::uniform_int_distribution<i16> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    i32 mock_value(std::mt19937 & eng, i32 min_bound, i32 max_bound){
        std::uniform_int_distribution<i32> dist {min_bound,max_bound};
        return dist(eng);
    }

    template<>
    i64 mock_value(std::mt19937 & eng, i64 min_bound, i64 max_bound){
        std::uniform_int_distribution<i64> dist {min_bound,max_bound};
        return dist(eng);
    }



    //mock vector
    template<>
    std::vector<u32> mock_vector(u64 seed,u32 len, u32 min_bound, u32 max_bound){
        std::vector<u32> vec;

        std::mt19937 eng{seed};

        for(u32 i = 0; i < len; i++){
            vec.push_back(mock_value(eng, min_bound, max_bound));
        }

        return std::move(vec);
    }

    template<>
    std::vector<u64> mock_vector(u64 seed,u32 len, u64 min_bound, u64 max_bound){
        std::vector<u64> vec;

        std::mt19937 eng{seed};

        for(u32 i = 0; i < len; i++){
            vec.push_back(mock_value(eng, min_bound, max_bound));
        }

        return std::move(vec);
    }




    template<> 
    sycl::buffer<u32> mock_buffer(u64 seed,u32 len, u32 min_bound, u32 max_bound){
        return shamalgs::memory::vec_to_buf(
            mock_vector(seed, len, min_bound, max_bound)
        );
    }




}