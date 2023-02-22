// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include <map>

namespace shamrock::scheduler {
    template<class T>
    class DistributedData {

        std::map<u64, T> data;

        public:

        void add_obj(u64 id , T && obj){
            data.emplace(id,obj);
        }

        template<class Fct>
        void for_each(Fct &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }

        T& get(u64 id){
            return data.at(id);
        }


    };
} // namespace shamrock::scheduler
