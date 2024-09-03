// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DistributedData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include <functional>
#include <map>
#include <utility>

namespace shambase {

    /**
     * @brief Represents a collection of objects distributed across patches identified by a u64 id.
     *
     * This class provides methods for managing the distributed data collection,
     * including adding, removing, finding, and iterating over elements.
     * It also supports mapping the collection to a new type using a user-defined mapping function.
     *
     * @tparam T The type of the object in the collection.
     */
    template<class T>
    class DistributedData {

        std::map<u64, T> data; ///< The underlying collection.

        using iterator = typename std::map<u64, T>::iterator; ///< Iterator type.

        public:
        /**
         * @brief Returns the underlying collection.
         *
         * @return The underlying collection.
         */
        inline std::map<u64, T> &get_native() { return data; }

        /**
         * @brief Adds a new object to the collection.
         *
         * @param id The id of the patch the object belongs to.
         * @param obj The object to add.
         *
         * @return An iterator pointing to the inserted object.
         *
         * @throw If the key already exist.
         */
        inline iterator add_obj(u64 id, T &&obj) {

            std::pair<iterator, bool> ret = data.emplace(id, std::forward<T>(obj));

            if (!ret.second) {
                throw make_except_with_loc<std::runtime_error>("the key already exist");
            }

            return ret.first;
        }

        /**
         * @brief Removes an object from the collection.
         *
         * @param id The id of the patch the object belongs to.
         */
        inline void erase(u64 id) { data.erase(id); }

        inline void for_each(std::function<void(u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id, obj);
            }
        }

        /**
         * @brief Finds an object in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return An iterator pointing to the object if found, or data.end() otherwise.
         */
        inline auto find(u64 id) { return data.find(id); }

        /**
         * @brief Returns an iterator pointing to the end of the collection.
         *
         * @return An iterator pointing to the end of the collection.
         */
        inline auto not_found() { return data.end(); }

        /**
         * @brief Returns a reference to an object in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return A reference to the object.
         *
         * @throw If the object is not found.
         */
        inline T &get(u64 id) {
            try {
                return data.at(id);
            } catch (std::out_of_range &) {

                std::vector<u64> id_list;

                for_each([&](u64 id, T &) {
                    id_list.push_back(id);
                });

                throw make_except_with_loc<std::runtime_error>(format(
                    "The querried id {} does not exist, current id list is {}", id, id_list));
            }
        }

        /**
         * @brief Checks if an object exists in the collection.
         *
         * @param id The id of the patch the object belongs to.
         *
         * @return True if the object is found, false otherwise.
         */
        inline bool has_key(u64 id) { return (data.find(id) != data.end()); }

        /**
         * @brief Returns the number of elements in the collection.
         *
         * @return The number of elements in the collection.
         */
        inline u64 get_element_count() { return data.size(); }

        /**
         * @brief Returns true if the collection is empty.
         *
         * @return True if the collection is empty, false otherwise.
         */
        inline bool is_empty() { return data.empty(); }

        /**
         * @brief Prints all the objects in the collection to the logger.
         *
         * The format string is passed to fmt::format to format each object, so
         * the syntax is the same as fmt::format. The object is passed as the
         * second argument to fmt::format.
         *
         * Example:
         * \code{.cpp}
         * DistributedData<int> data;
         * data.add_obj(0, 42);
         * data.add_obj(1, 24);
         * data.print_data("{:d}"); // will print "0 -> 42" and "1 -> 24"
         * \endcode
         *
         * @tparam Tf Types of the format string placeholders.
         * @param fmt The format string.
         */
        template<typename... Tf>
        inline void print_data(fmt::format_string<Tf...> fmt) {
            for_each([&](u64 id_patch, T &ref) {
                logger::raw_ln(id_patch, "->", shambase::format(fmt, ref));
            });
        }

        /**
         * @brief Apply a function to all objects in the collection and return
         *        a new collection containing the results.
         *
         * The `map()` function applies the given function to each object in the
         * collection and stores the result in a new collection. The function
         * is passed the id of the object and a reference to the object as
         * arguments.
         *
         * Example:
         * \code{.cpp}
         * DistributedData<int> data;
         * data.add_obj(0, 42);
         * data.add_obj(1, 24);
         *
         * // a DistributedData<float> with doubled values from the input
         * auto mapped = data.map<float>([](u64 id, int& val) { return val * 2.; });
         * \endcode
         *
         * @tparam Tmap Type of the objects in the returned collection.
         * @param map_func The function to apply to each object.
         * @return A new collection containing the results of the map function.
         */
        template<class Tmap>
        inline DistributedData<Tmap> map(std::function<Tmap(u64, T &)> map_func) {
            DistributedData<Tmap> ret;

            for_each([&](u64 id, T &ref) {
                ret.add_obj(id, map_func(id, ref));
            });

            return ret;
        }

        /**
         * @brief Reset the collection to its initial state
         *
         * All objects in the collection are removed.
         */
        inline void reset() { data.clear(); }
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
        inline std::multimap<std::pair<u64, u64>, T> &get_native() { return data; }

        inline iterator add_obj(u64 left_id, u64 right_id, T &&obj) {
            std::pair<u64, u64> tmp = {left_id, right_id};
            return data.emplace(std::move(tmp), std::forward<T>(obj));
        }

        inline void for_each(std::function<void(u64, u64, T &)> &&f) {
            for (auto &[id, obj] : data) {
                f(id.first, id.second, obj);
            }
        }

        inline void tranfer_all(std::function<bool(u64, u64)> cd, DistributedDataShared &other) {

            std::vector<std::pair<u64, u64>> occurences;

            // whoa i forgot the & here and triggered the copy constructor of every patch
            // like do not forget it or it will be a disaster waiting to come
            // i did throw up a 64 GPUs run because of that
            for (auto &[k, v] : data) {
                if (cd(k.first, k.second)) {
                    occurences.push_back(k);
                }
            }

            for (auto p : occurences) {
                auto ext = data.extract(p);
                other.data.insert(std::move(ext));
            }
        }

        inline bool has_key(u64 left_id, u64 right_id) {
            return (data.find({left_id, right_id}) != data.end());
        }

        inline u64 get_element_count() { return data.size(); }

        template<class Tmap>
        inline DistributedDataShared<Tmap> map(std::function<Tmap(u64, u64, T &)> map_func) {
            DistributedDataShared<Tmap> ret;
            for_each([&](u64 left, u64 right, T &ref) {
                ret.add_obj(left, right, map_func(left, right, ref));
            });
            return ret;
        }

        inline void reset() { data.clear(); }

        inline bool is_empty() { return data.empty(); }
    };
} // namespace shambase
