// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "patchdata_field.hpp"
#include "CL/sycl/access/access.hpp"
#include "CL/sycl/accessor.hpp"
#include "core/patch/base/enabled_fields.hpp"
#include "core/patch/base/pdat_comm_impl/pdat_comm_cp_to_host.hpp"
#include "core/patch/base/pdat_comm_impl/pdat_comm_directgpu.hpp"

//TODO use hash for name + nvar to check if the field match before doing operation on them

template <class T> void PatchDataField<T>::extract_element(u32 pidx, PatchDataField<T> &to) {

    auto fast_extract_ptr = [](u32 idx, u32 lenght, auto cnt) {
        T end_ = cnt[lenght - 1];
        T extr = cnt[idx];

        cnt[idx] = end_;

        return extr;
    };

    auto sub_extract = [fast_extract_ptr](u32 pidx, PatchDataField<T> &from, PatchDataField<T> &to) {
        const u32 nvar        = from.get_nvar();
        const u32 idx_val     = pidx * nvar;
        const u32 idx_out_val = to.size();

        u32 from_sz = from.size();

        to.expand(1);

        {
            auto buf_to   = to.get_sub_buf();
            auto buf_from = from.get_sub_buf();

            sycl::host_accessor acc_to{*buf_to};
            sycl::host_accessor acc_from{*buf_from};

            for (u32 i = nvar - 1; i < nvar; i--) {
                acc_to[idx_out_val + i] = (fast_extract_ptr(idx_val + i, from_sz, acc_from));
            }
        }

        from.shrink(1);
    };

    sub_extract(pidx, *this, to);
}

template <class T> bool PatchDataField<T>::check_field_match(PatchDataField<T> &f2) {
    bool match = true;

    match = match && (field_name == f2.field_name);
    match = match && (nvar == f2.nvar);
    match = match && (obj_cnt == f2.obj_cnt);
    match = match && (val_cnt == f2.val_cnt);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf};

        auto buf_f2 = f2.get_sub_buf();
        sycl::host_accessor acc_f2{*buf};

        for (u32 i = 0; i < val_cnt; i++) {
            match = match && test_sycl_eq(acc[i], acc_f2[i]);
        }
    }

    return match;
}

template <class T> void PatchDataField<T>::append_subset_to(std::vector<u32> &idxs, PatchDataField &pfield) {

    if (pfield.nvar != nvar)
        throw shamrock_exc("field must be similar for extraction");

    const u32 start_enque = pfield.val_cnt;

    const u32 nvar = get_nvar();

    pfield.expand(idxs.size());

    {

        auto buf_curr  = get_sub_buf();
        auto buf_other  = pfield.get_sub_buf();

        sycl::host_accessor acc_curr{*buf_curr};
        sycl::host_accessor acc_other{*buf_other};

        for (u32 i = 0; i < idxs.size(); i++) {

            const u32 idx_extr = idxs[i] * nvar;
            const u32 idx_push = start_enque + i * nvar;

            for (u32 a = 0; a < nvar; a++) {
                acc_other[idx_push + a] = acc_curr[idx_extr + a];
            }
        }
        
    }
    // auto buf_cur  = data();
    // auto buf_other  = data();
    //
    //{
    //     sycl::host_accessor acc_cur{*buf_cur};
    //     sycl::host_accessor acc_other{*buf_cur};
    //
    //     for (u32 i = 0; i < idxs.size(); i++) {
    //
    //         const u32 idx_extr = idxs[i]*nvar;
    //         const u32 idx_push = start_enque + i*nvar;
    //
    //         for(u32 a = 0; a < nvar ; a++){
    //             acc_other[idx_push + a] = acc_cur[idx_extr + a];
    //         }
    //
    //     }
    // }
}

#define X(a) template class PatchDataField<a>;
XMAC_LIST_ENABLED_FIELD
#undef X

template <> void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32(distf64(eng));
        }
    }

    // for (auto & a : field_data) {
    //     a = f32(distf64(eng));
    // }
}

template <> void PatchDataField<f32_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_2{distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_3{distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_8{distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                           distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_16{distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64(distf64(eng));
        }
    }
}

template <> void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_2{distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_3{distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_8{distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                           distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_16{distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = distu32(eng);
        }
    }
}
template <> void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1, 6000);

    {
        auto buf = get_sub_buf();
        sycl::host_accessor acc{*buf, sycl::write_only};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = distu64(eng);
        }
    }
}

namespace patchdata_field {

    comm_type current_mode = CopyToHost;

    template <class T>
    PatchDataFieldMpiRequest<T>::PatchDataFieldMpiRequest(PatchDataField<T> &pdat_field, comm_type comm_mode,
                                                          op_type comm_op, u32 comm_sz)
        : comm_mode(comm_mode), comm_op(comm_op), comm_sz(comm_sz), pdat_field(pdat_field) {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "starting mpi sycl comm ", comm_sz, comm_op, comm_mode);

        if (comm_mode == CopyToHost && comm_op == Send) {

            comm_ptr = impl::copy_to_host::send::init<T>(pdat_field, comm_sz);

        } else if (comm_mode == CopyToHost && comm_op == Recv) {

            comm_ptr = impl::copy_to_host::recv::init<T>(comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            comm_ptr = impl::directgpu::send::init<T>(pdat_field, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Recv) {

            comm_ptr = impl::directgpu::recv::init<T>(comm_sz);

        } else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

    template <class T> void PatchDataFieldMpiRequest<T>::finalize() {

        logger::debug_mpi_ln("PatchDataField MPI Comm", "finalizing mpi sycl comm ", comm_sz, comm_op, comm_mode);

        pdat_field.resize(comm_sz);

        if (comm_mode == CopyToHost && comm_op == Send) {

            impl::copy_to_host::send::finalize<T>(comm_ptr);

        } else if (comm_mode == CopyToHost && comm_op == Recv) {

            impl::copy_to_host::recv::finalize<T>(pdat_field, comm_ptr, comm_sz);

        } else if (comm_mode == DirectGPU && comm_op == Send) {

            impl::directgpu::send::finalize<T>(comm_ptr);

        } else if (comm_mode == DirectGPU && comm_op == Recv) {

            impl::directgpu::recv::finalize<T>(pdat_field, comm_ptr, comm_sz);

        } else {
            logger::err_ln("PatchDataField MPI Comm", "communication mode & op combination not implemented :", comm_mode,
                           comm_op);
        }
    }

#define X(a) template struct PatchDataFieldMpiRequest<a>;
    XMAC_LIST_ENABLED_FIELD
#undef X

} // namespace patchdata_field
