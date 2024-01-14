// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file serialize.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "details/SerializeHelperMember.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace shamalgs {

    namespace details {
        template<u64 alignment>
        inline u64 align_repr(u64 offset) {
            u64 modval = offset % alignment;
            if (modval == 0) {
                return offset;
            }
            return offset + (alignment - modval);
        }

        template<u64 alignment, class T>
        inline u64 serialize_byte_size() {
            using Helper = details::SerializeHelperMember<T>;
            return align_repr<alignment>(Helper::szrepr);
        }

        template<u64 alignment, class T>
        inline u64 serialize_byte_size(u64 len) {
            using Helper = details::SerializeHelperMember<T>;
            return align_repr<alignment>(len * Helper::szrepr);
        }

        template<u64 alignment>
        inline u64 serialize_byte_size(std::string s) {
            return serialize_byte_size<alignment, u32>() +
                   serialize_byte_size<alignment, u32>(s.size());
        }

    } // namespace details


    struct SerializeSize{
        u64 head_size = 0;
        u64 content_size = 0;

        SerializeSize& operator+=(const SerializeSize& rhs) // compound assignment (does not need to be a member,
        {                           // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size += rhs.head_size;
            content_size += rhs.content_size;
            return *this; // return the result by reference
        }
    
        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator+(SerializeSize lhs,        // passing lhs by value helps optimize chained a+b+c
                        const SerializeSize& rhs) // otherwise, both parameters may be const references
        {
            lhs += rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }
        SerializeSize& operator*=(const SerializeSize& rhs) // compound assignment (does not need to be a member,
        {                           // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size *= rhs.head_size;
            content_size *= rhs.content_size;
            return *this; // return the result by reference
        }
    
        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator*(SerializeSize lhs,        // passing lhs by value helps optimize chained a*b*c
                        const SerializeSize& rhs) // otherwise, both parameters may be const references
        {
            lhs *= rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }


        SerializeSize& operator*=(const int& rhs) // compound assignment (does not need to be a member,
        {                           // but often is, to modify the private members)
            /* addition of rhs to *this takes place here */
            head_size *= rhs;
            content_size *= rhs;
            return *this; // return the result by reference
        }
    
        // friends defined inside class body are inline and are hidden from non-ADL lookup
        friend SerializeSize operator*(SerializeSize lhs,        // passing lhs by value helps optimize chained a*b*c
                        const int& rhs) // otherwise, both parameters may be const references
        {
            lhs *= rhs; // reuse compound assignment
            return lhs; // return the result by value (uses move constructor)
        }

        static SerializeSize Header(u64 sz){
            return {sz,0};
        }
        static SerializeSize Content(u64 sz){
            return {0,sz};
        }

        inline u64 get_total_size(){
            return head_size + content_size;
        }

    };

    class SerializeHelper {
        std::unique_ptr<sycl::buffer<u8>> storage;
        u64 head      = 0;
        bool first_op = true;

        static constexpr u64 alignment = 8;

        inline void check_head_move(u64 off) {
            if (!storage) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    ("the buffer is not allocated, the head cannot be moved"));
            }
            if (head + off > storage->size()) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    shambase::format("the buffer is not allocated, the head cannot be moved\n "
                                     "storage size : {}, requested head : {}",
                                     storage->size(),
                                     head + off));
            }
        }

        inline static u64 align_repr(u64 offset) { return details::align_repr<alignment>(offset); }

        public:
        SerializeHelper() = default;

        SerializeHelper(std::unique_ptr<sycl::buffer<u8>> &&storage)
            : storage(std::forward<std::unique_ptr<sycl::buffer<u8>>>(storage)) {}

        inline void allocate(SerializeSize szinfo) {
            u64 bytelen = szinfo.head_size + szinfo.content_size;
            StackEntry stack_loc{false};
            storage  = std::make_unique<sycl::buffer<u8>>(bytelen);
            head     = 0;
            first_op = true;
        }

        inline std::unique_ptr<sycl::buffer<u8>> finalize() {
            StackEntry stack_loc{false};
            std::unique_ptr<sycl::buffer<u8>> ret;
            std::swap(ret, storage);
            return ret;
        }

        template<class T>
        inline static SerializeSize serialize_byte_size() {
            return SerializeSize::Header(details::serialize_byte_size<alignment, T>());
        }

        template<class T>
        inline static SerializeSize serialize_byte_size(u64 len) {
            return SerializeSize::Content(details::serialize_byte_size<alignment, T>(len));
        }

        inline static SerializeSize serialize_byte_size(std::string s) {
            return SerializeSize::Header(details::serialize_byte_size<alignment>(s));
        }

        template<class T>
        inline void write(T val) {
            StackEntry stack_loc{false};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head;

            u64 offset = align_repr(Helper::szrepr);
            check_head_move(offset);

            if (first_op) {
                shamsys::instance::get_compute_queue().submit(
                    [&, val, current_head](sycl::handler &cgh) {
                        sycl::accessor accbuf{*storage, cgh, sycl::write_only, sycl::no_init};
                        cgh.single_task([=]() { Helper::store(&accbuf[current_head], val); });
                    });
                first_op = false;
            } else {
                shamsys::instance::get_compute_queue().submit(
                    [&, val, current_head](sycl::handler &cgh) {
                        sycl::accessor accbuf{*storage, cgh, sycl::write_only};
                        cgh.single_task([=]() { Helper::store(&accbuf[current_head], val); });
                    });
            }

            head += offset;
        }

        template<class T>
        inline void load(T &val) {
            StackEntry stack_loc{false};

            using Helper = details::SerializeHelperMember<T>;

            u64 current_head = head;
            u64 offset       = align_repr(Helper::szrepr);
            check_head_move(offset);

            {//using host_acc rather than anything else since other options causes addition latency
                sycl::host_accessor accbuf{*storage, sycl::read_only};
                val = Helper::load(&accbuf[current_head]);
            }

            head += offset;
        }

        template<class T>
        inline void write_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move(offset);

            if (first_op) {
                shamsys::instance::get_compute_queue().submit(
                    [&, current_head](sycl::handler &cgh) {
                        sycl::accessor accbufbyte{*storage, cgh, sycl::write_only, sycl::no_init};
                        sycl::accessor accbuf{buf, cgh, sycl::read_only};

                        cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                            u64 head = current_head + id.get_linear_id() * Helper::szrepr;
                            Helper::store(&accbufbyte[head], accbuf[id]);
                        });
                    });
                first_op = false;
            } else {
                shamsys::instance::get_compute_queue().submit(
                    [&, current_head](sycl::handler &cgh) {
                        sycl::accessor accbufbyte{*storage, cgh, sycl::write_only};
                        sycl::accessor accbuf{buf, cgh, sycl::read_only};

                        cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                            u64 head = current_head + id.get_linear_id() * Helper::szrepr;
                            Helper::store(&accbufbyte[head], accbuf[id]);
                        });
                    });
            }

            head += offset;
        }

        template<class T>
        inline void load_buf(sycl::buffer<T> &buf, u64 len) {
            StackEntry stack_loc{false};

            using Helper     = details::SerializeHelperMember<T>;
            u64 current_head = head;

            u64 offset = align_repr(len * Helper::szrepr);
            check_head_move(offset);

            shamsys::instance::get_compute_queue().submit([&, current_head](sycl::handler &cgh) {
                sycl::accessor accbufbyte{*storage, cgh, sycl::read_only};
                sycl::accessor accbuf{buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                    u64 head   = current_head + id.get_linear_id() * Helper::szrepr;
                    accbuf[id] = Helper::load(&accbufbyte[head]);
                });
            });

            head += offset;
        }

        inline void write(std::string s) {
            StackEntry stack_loc{false};
            write(u32(s.size()));

            sycl::buffer<char> buf(s.size());
            {
                sycl::host_accessor acc{buf, sycl::write_only, sycl::no_init};
                for (u32 i = 0; i < s.size(); i++) {
                    acc[i] = s[i];
                }
            }
            write_buf(buf, s.size());
        }

        inline void load(std::string &s) {
            StackEntry stack_loc{false};
            u32 len;
            load(len);
            s.resize(len);

            sycl::buffer<char> buf(len);
            load_buf(buf, len);
            {
                sycl::host_accessor acc{buf, sycl::read_only};
                for (u32 i = 0; i < len; i++) {
                    s[i] = acc[i];
                }
            }
        }
    };

} // namespace shamalgs