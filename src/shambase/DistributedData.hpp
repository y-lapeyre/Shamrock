// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/exception.hpp"
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

        using iterator = typename std::map<u64, T>::iterator;

        public:
        inline std::map<u64, T> &get_native() { return data; }

        inline iterator add_obj(u64 id, T &&obj) {

            std::pair<iterator, bool> ret = data.emplace(id, std::forward<T>(obj));

            if (!ret.second) {
                throw throw_with_loc<std::runtime_error>("the key already exist");
            }

            return ret.first;
        }

        inline void erase(u64 id) { data.erase(id); }

        inline void for_each(std::function<void(u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }
        inline auto find(u64 id) { return data.find(id); }

        inline auto not_found() { return data.end(); }

        inline T &get(u64 id) { return data.at(id); }

        inline bool has_key(u64 id) { return (data.find(id) != data.end()); }

        inline u64 get_element_count() { return data.size(); }
    };

    /**
     * @brief Describe an object common to two patches, typically interface (sender,receiver)
     *
     * @tparam T
     */
    template<class T>
    class DistributedDataShared {

        std::multimap<std::pair<u64, u64>, T> data;

        using iterator = typename std::multimap<std::pair<u64, u64>, T>::iterator;

        public:
        inline std::map<std::pair<u64, u64>, T> &get_native() { return data; }

        inline iterator add_obj(u64 left_id, u64 right_id, T &&obj) {
            std::pair<u64, u64> tmp = {left_id, right_id};
            return data.emplace(std::move(tmp), std::forward<T>(obj));
        }

        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        inline bool has_key(u64 left_id, u64 right_id) {
            return (data.find({left_id, right_id}) != data.end());
        }

        inline u64 get_element_count() { return data.size(); }
    };
} // namespace shambase
