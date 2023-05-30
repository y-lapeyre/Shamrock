// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamalgs/numeric/numeric.hpp"
#include "shambase/DistributedData.hpp"
#include "shambase/sycl.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversal.hpp"

namespace shamrock::tree {

    class ObjectCacheHandler {

        shambase::DistributedData<ObjectCache> cache;

        std::function<ObjectCache(u64)> generator;

        std::deque<u64> last_builds;

        u64 max_memsize;
        u64 current_memsize = 0;

        public:
        ObjectCacheHandler(u64 max_memsize, std::function<ObjectCache(u64)> &&generator)
            : max_memsize(max_memsize), generator(generator) {}

        /**
         * @brief pop oldest entry in the cache
         *
         */
        inline void pop_oldest() {

            bool successfull_pop = false;
            do {

                if (cache.is_empty()) {
                    throw shambase::throw_with_loc<std::runtime_error>(
                        "the cache is empty no entry can be popped");
                }

                u64 last_id = last_builds.front();
                last_builds.pop_front();

                if (cache.has_key(last_id)) {
                    current_memsize -= cache.get(last_id).get_memsize();
                    cache.erase(last_id);
                    successfull_pop = true;
                    logger::debug_ln("ObjectCacheHandler",
                                     "deleted cache for id =",
                                     last_id,
                                     "cachesize =",
                                     shambase::readable_sizeof(current_memsize));
                }

            } while (!successfull_pop);
        }

        /**
         * @brief push a new entry in the cache, and check that enough space is available
         *
         * @param id
         * @param c
         */
        inline void push_new(u64 id, ObjectCache &&c) {
            u64 add_sz = c.get_memsize();

            if (add_sz + current_memsize > max_memsize) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "their is no space within the imposed limit, try freeing some space in the "
                    "cache, or increase the size limit");
            }

            current_memsize += add_sz;
            cache.add_obj(id, std::forward<ObjectCache>(c));
            last_builds.push_back(id);
        }

        /**
         * @brief build a new entry in the cache and free enough
         * entry to make space for it if necessary
         *
         * @param id
         */
        inline void build_cache(u64 id) {
            ObjectCache new_cache = generator(id);

            u64 new_sz = new_cache.get_memsize();

            while (new_sz + current_memsize > max_memsize) {
                if (cache.is_empty()) {
                    throw shambase::throw_with_loc<std::runtime_error>(
                        "their no space left to allocate a cache, try with smaller objects, or "
                        "increase the size limit");
                }
                logger::warn_ln("ObjectCacheHandler",
                                "The cache is too small, or the objects too big, some caches will "
                                "have to be recomputed on the fly");
                pop_oldest();
            }

            push_new(id, std::move(new_cache));
            logger::debug_ln("ObjectCacheHandler",
                             "built cache for id =",
                             id,
                             "cachesize =",
                             shambase::readable_sizeof(current_memsize));
        }

        /**
         * @brief Get a cache entry and build it if it is not already in the cache
         *
         * @param id
         * @return ObjectCache&
         */
        inline ObjectCache &get_cache(u64 id) {
            if (cache.has_key(id)) {
                return cache.get(id);
            } else {
                build_cache(id);
                return cache.get(id);
            }
        }

        inline void reset() {
            last_builds.clear();
            cache.reset();
            current_memsize = 0;
        }
    };

} // namespace shamrock::tree
