// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include <map>
#include <utility>

namespace shambase {

    /**
     * @brief Describe an object distributed accros patches (id = u64)
     * 
     * @tparam T 
     */
    template<class T>
    class DistributedData {

        std::map<u64, T> data;

        public:
        inline std::map<u64, T> &get_native() { return data; }

        inline void add_obj(u64 id, T &&obj) { data.emplace(id, std::forward<T>(obj)); }

        inline void for_each(std::function<void(u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }

        T &get(u64 id) { return data.at(id); }

        bool has_key(u64 id) { return (data.find(id) != data.end()); }

        u64 get_element_count() { return data.size(); }
    };

    /**
     * @brief Describe an object common to two patches, typically interface (sender,receiver)
     * 
     * @tparam T 
     */
    template<class T>
    class DistributedDataShared {

        std::map<std::pair<u64, u64>, T> data;

        public:
        inline std::map<std::pair<u64, u64>, T> &get_native() { return data; }

        inline void add_obj(u64 left_id, u64 right_id, T &&obj) {
            std::pair<u64,u64> tmp = {left_id, right_id};
            data.emplace(std::move(tmp), std::forward<T>(obj));
        }

        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        T &get(u64 left_id, u64 right_id) { return data.at({left_id, right_id}); }

        bool has_key(u64 left_id, u64 right_id) {
            return (data.find({left_id, right_id}) != data.end());
        }

        u64 get_element_count() { return data.size(); }
    };
} // namespace shambase
