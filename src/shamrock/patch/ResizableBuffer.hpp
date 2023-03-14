// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#pragma once

#include "aliases.hpp"
#include "shamalgs/algorithm/algorithm.hpp"
#include "shamrock/legacy/algs/sycl/sycl_algs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

template<class T>
class ResizableBuffer {

    
    std::unique_ptr<sycl::buffer<T>> buf;

    u32 capacity;
    u32 val_cnt; // nvar*obj_cnt

    constexpr static u32 min_capa  = 100;
    constexpr static f32 safe_fact = 1.25;

    void alloc() {
        buf = std::make_unique<sycl::buffer<T>>(capacity);

        logger::debug_alloc_ln("PatchDataField", "allocate field :", "len =", capacity);
    }

    void free() {

        if (buf) {
            logger::debug_alloc_ln("PatchDataField", "free field :", "len =", capacity);

            buf.reset();
        }
    }





    public:












    ////////////
    // getters
    ////////////
    
    /**
     * @brief Get the reference to the internal buffer
     * 
     * @return const std::unique_ptr<sycl::buffer<T>>& 
     */
    inline const std::unique_ptr<sycl::buffer<T>> &get_buf() const { return buf; }

    inline std::unique_ptr<sycl::buffer<T>> &get_buf_priviledge() { return buf; }

    [[nodiscard]] inline const u32 &size() const { return val_cnt; }

    [[nodiscard]] inline u64 memsize() const { return val_cnt * sizeof(T); }





    inline void resize(u32 new_size) {
        logger::debug_alloc_ln("ResizableBuffer", "resize from : ", val_cnt, "to :", new_size);

        if (capacity == 0) {
            capacity = safe_fact * new_size;
            alloc();
        } else if (new_size > capacity) {

            // u32 old_capa = capacity;
            capacity = safe_fact * new_size;

            sycl::buffer<T> *old_buf = buf.release();

            alloc();

            syclalgs::basic::copybuf_discard(*old_buf, *buf, val_cnt);

            logger::debug_alloc_ln("PatchDataField", "delete old buf : ");
            delete old_buf;
        } else {
        }

        val_cnt = new_size;
    }

    inline void overwrite(ResizableBuffer<T> &f2, u32 cnt) {
        if (val_cnt < cnt) {
            throw shambase::throw_with_loc<std::invalid_argument>(
                "to overwrite you need more element in the field"
            );
        }

        {
            sycl::host_accessor acc{*buf};
            sycl::host_accessor acc_f2{*f2.get_buf()};

            for (u32 i = 0; i < cnt; i++) {
                // field_data[idx_st + i] = f2.field_data[i];
                acc[i] = acc_f2[i];
            }
        }
    }

    inline void override(sycl::buffer<T> &data, u32 cnt) {

        if (cnt != val_cnt)
            throw shambase::throw_with_loc<std::invalid_argument>(
                "buffer size doesn't match patchdata field size"
            ); // TODO remove ref to size

        if (val_cnt > 0) {

            {
                sycl::host_accessor acc_cur{*buf};
                sycl::host_accessor acc{data, sycl::read_only};

                for (u32 i = 0; i < val_cnt; i++) {
                    // field_data[i] = acc[i];
                    acc_cur[i] = acc[i];
                }
            }
        }
    }

    inline void override(const T val) {

        if (val_cnt > 0) {

            {
                sycl::host_accessor acc{*buf};
                for (u32 i = 0; i < val_cnt; i++) {
                    // field_data[i] = val;
                    acc[i] = val;
                }
            }
        }
    }

    static sycl::buffer<T> convert_to_buf(ResizableBuffer<T> && rbuf){
        std::unique_ptr<sycl::buffer<T>> buf_recov;
        
        std::swap(rbuf.buf, buf_recov);

        sycl::buffer<T>* ptr = buf_recov.release();

        return *ptr;
    }

    inline void index_remap_resize(sycl::buffer<u32> & index_map, u32 len, u32 nvar = 1){
        if(get_buf()){

            auto get_new_buf = [&](){
                if(nvar == 1){
                    return shamalgs::algorithm::index_remap(
                        shamsys::instance::get_compute_queue(), 
                        *get_buf(), 
                        index_map, 
                        len);
                }else{
                    return shamalgs::algorithm::index_remap_nvar(
                        shamsys::instance::get_compute_queue(), 
                        *get_buf(), 
                        index_map, 
                        len, nvar);
                }
            };

            sycl::buffer<T> new_buf = get_new_buf();

            capacity = new_buf.size();
            val_cnt = len*nvar;
            buf = std::make_unique<sycl::buffer<T>>(std::move(new_buf));
        }
    }



    explicit ResizableBuffer(u32 cnt = 0) : val_cnt(cnt), capacity(cnt) {
        if (capacity != 0) {
            alloc();
        }
    }

    ResizableBuffer(const ResizableBuffer &other)
        : val_cnt(other.val_cnt), capacity(other.capacity) {
        if (capacity != 0) {
            alloc();
            // copydata(other._data,_data, capacity);
            syclalgs::basic::copybuf_discard(*other.buf, *buf, capacity);
        }
    }

    inline ResizableBuffer(sycl::buffer<T> &&moved_buf, u32 val_cnt)
        : val_cnt(val_cnt), capacity(moved_buf.size()) {
        buf = std::make_unique<sycl::buffer<T>>(std::move(moved_buf));
    }

    

    ResizableBuffer(ResizableBuffer &&other) noexcept
        : buf(std::move(other.buf)), val_cnt(std::move(other.val_cnt)),
          capacity(std::move(other.capacity)) {} // move constructor

    ResizableBuffer &operator=(ResizableBuffer &&other) noexcept {
        buf      = std::move(other.buf);
        val_cnt  = std::move(other.val_cnt);
        capacity = std::move(other.capacity);

        return *this;
    } // move assignment

    ResizableBuffer& operator=(const ResizableBuffer& other) // copy assignment
     = delete;


    inline ~ResizableBuffer() {
        logger::debug_alloc_ln("ResizableBuffer", "free field :", "len =", capacity);
        free();
    }
};
