// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "details/SerializeHelperMember.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/sycl.hpp"
#include "shambase/type_aliases.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace shamalgs {

    class SerializeHelper {
        std::unique_ptr<sycl::buffer<u8>> storage;
        u64 head = 0;

        inline void check_head_move(u64 off){
            if(!storage){
                throw shambase::throw_with_loc<std::runtime_error>("the buffer is not allocated, the head cannot be moved");
            }
            if(head + off > storage->size()){
                throw shambase::throw_with_loc<std::runtime_error>("this operation can not move the head passed the size of the storage");
            }
        }

        public:

        SerializeHelper() = default;

        SerializeHelper(std::unique_ptr<sycl::buffer<u8>> && storage):
        storage(std::forward<std::unique_ptr<sycl::buffer<u8>>>(storage)){}

        inline void allocate(u64 bytelen) {
            StackEntry stack_loc{};
            storage = std::make_unique<sycl::buffer<u8>>(bytelen);
            head    = 0;
        }

        inline std::unique_ptr<sycl::buffer<u8>> finalize() {
            StackEntry stack_loc{};
            std::unique_ptr<sycl::buffer<u8>> ret;
            std::swap(ret, storage);
            return ret;
        }

        template<class T>
        inline void write(T val) {
            StackEntry stack_loc{};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head;

            check_head_move(Helper::szrepr);

            shamsys::instance::get_compute_queue().submit(
                [&, val, current_head](sycl::handler &cgh) {
                    sycl::accessor accbuf{*storage, cgh, sycl::write_only};
                    cgh.single_task([=]() { Helper::store(&accbuf[current_head], val); });
                });

            head += Helper::szrepr;
        }

        template<class T>
        inline void load(T &val) {
            StackEntry stack_loc{};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head;
            check_head_move(Helper::szrepr);

            sycl::buffer<T> retbuf(1);

            shamsys::instance::get_compute_queue().submit([&, current_head](sycl::handler &cgh) {
                sycl::accessor accbuf{*storage, cgh, sycl::read_only};
                sycl::accessor retacc{retbuf, cgh, sycl::write_only, sycl::no_init};
                cgh.single_task(
                    [=]() { retacc[0] = Helper::load(&accbuf[current_head]); });
            });

            {
                sycl::host_accessor acc{retbuf, sycl::read_only};
                val = acc[0];
            }

            head += Helper::szrepr;
        }

        template<class T>
        inline void write_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head;


            check_head_move(len * Helper::szrepr);

            shamsys::instance::get_compute_queue().submit([&, current_head](sycl::handler &cgh) {
                sycl::accessor accbufbyte{*storage, cgh, sycl::write_only};
                sycl::accessor accbuf{buf, cgh, sycl::read_only};

                cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                    u64 head = current_head + id.get_linear_id() * Helper::szrepr;
                    Helper::store(&accbufbyte[head], accbuf[id]);
                });
            });

            head += len * Helper::szrepr;
        }

        template<class T>
        inline void load_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head;

            check_head_move(len * Helper::szrepr);

            shamsys::instance::get_compute_queue().submit([&, current_head](sycl::handler &cgh) {
                sycl::accessor accbufbyte{*storage, cgh, sycl::read_only};
                sycl::accessor accbuf{buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                    u64 head   = current_head + id.get_linear_id() * Helper::szrepr;
                    accbuf[id] = Helper::load(&accbufbyte[head]);
                });
            });

            head += len * Helper::szrepr;
        }
    };

} // namespace shamalgs