// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "ResizableBuffer.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////
// memory manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void ResizableBuffer<T>::alloc() {
    buf = std::make_unique<sycl::buffer<T>>(capacity);

    logger::debug_alloc_ln("PatchDataField", "allocate field :", "len =", capacity);
}

template<class T>
void ResizableBuffer<T>::free() {

    if (buf) {
        logger::debug_alloc_ln("PatchDataField", "free field :", "len =", capacity);

        buf.reset();
    }
}

template<class T>
void ResizableBuffer<T>::resize(u32 new_size) {
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

template<class T>
sycl::buffer<T> ResizableBuffer<T>::convert_to_buf(ResizableBuffer<T> &&rbuf) {
    std::unique_ptr<sycl::buffer<T>> buf_recov;

    std::swap(rbuf.buf, buf_recov);

    sycl::buffer<T> *ptr = buf_recov.release();

    return *ptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// value manipulation
////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void ResizableBuffer<T>::overwrite(ResizableBuffer<T> &f2, u32 cnt) {
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

template<class T>
void ResizableBuffer<T>::override(sycl::buffer<T> &data, u32 cnt) {

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

template<class T>
void ResizableBuffer<T>::override(const T val) {

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

template<class T>
void ResizableBuffer<T>::index_remap_resize(sycl::buffer<u32> &index_map, u32 len, u32 nvar) {
    if (get_buf()) {

        auto get_new_buf = [&]() {
            if (nvar == 1) {
                return shamalgs::algorithm::index_remap(
                    shamsys::instance::get_compute_queue(), *get_buf(), index_map, len
                );
            } else {
                return shamalgs::algorithm::index_remap_nvar(
                    shamsys::instance::get_compute_queue(), *get_buf(), index_map, len, nvar
                );
            }
        };

        sycl::buffer<T> new_buf = get_new_buf();

        capacity = new_buf.size();
        val_cnt  = len * nvar;
        buf      = std::make_unique<sycl::buffer<T>>(std::move(new_buf));
    }
}

//////////////////////////////////////////////////////////////////////////
// Define the patchdata field for all classes in XMAC_LIST_ENABLED_FIELD
//////////////////////////////////////////////////////////////////////////

#define X(a) template class ResizableBuffer<a>;
XMAC_LIST_ENABLED_FIELD
#undef X

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////