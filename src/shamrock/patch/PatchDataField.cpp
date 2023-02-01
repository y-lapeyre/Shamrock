// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "PatchDataField.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"

template<class T> class Kernel_Extract_element;

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
            
            auto & buf_to   = to.get_buf();
            auto & buf_from = from.get_buf();

            shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

                sycl::accessor acc_to {*buf_to, cgh, sycl::write_only};
                sycl::accessor acc_from {*buf_from, cgh, sycl::read_write};

                const u32 nvar_loc = nvar;
                
                cgh.single_task<Kernel_Extract_element<T>>( [=](){
                    for (u32 i = nvar_loc - 1; i < nvar_loc; i--) {
                        acc_to[idx_out_val + i] = (fast_extract_ptr(idx_val + i, from_sz, acc_from));
                    }
                });

            });

        }

        from.shrink(1);
    };

    sub_extract(pidx, *this, to);
}




template<class T> class PdatField_checkfieldmatch;

template <class T> bool PatchDataField<T>::check_field_match(const PatchDataField<T> &f2) const {
    bool match = true;

    match = match && (field_name == f2.field_name);
    match = match && (nvar == f2.nvar);
    match = match && (obj_cnt == f2.obj_cnt);
    match = match && (val_cnt == f2.val_cnt);

    const u32 & check_len = val_cnt;

    {

        using buf_t = std::unique_ptr<sycl::buffer<T>>;

        const buf_t & buf = get_buf();
        const buf_t & buf_f2 = f2.get_buf();
        
        sycl::buffer<u8> res_buf(check_len);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc1 {*buf, cgh, sycl::read_only};
            sycl::accessor acc2 {*buf_f2, cgh, sycl::read_only};

            sycl::accessor acc_res {res_buf, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<PdatField_checkfieldmatch<T>>(sycl::range<1>{check_len}, [=](sycl::item<1> i){
                acc_res[i] = test_sycl_eq(acc1[i] , acc2[i]);
            });

        });
        
        match = match && syclalgs::reduction::is_all_true(res_buf,val_cnt);

    }

    return match;
}



template<class T> class PdatField_append_subset_to;

template <class T> void PatchDataField<T>::append_subset_to(sycl::buffer<u32> &idxs_buf,u32 sz, PatchDataField &pfield) const {

    if (pfield.nvar != nvar)
        throw shamrock_exc("field must be similar for extraction");

    const u32 start_enque = pfield.val_cnt;

    const u32 nvar = get_nvar();

    pfield.expand(sz);

    



    using buf_t = std::unique_ptr<sycl::buffer<T>>;

    const buf_t & buf_other  = pfield.get_buf();


    #ifdef false
    {
        sycl::host_accessor acc_idxs {idxs_buf, sycl::read_only};
        sycl::host_accessor acc_curr {*buf, sycl::read_only};
        sycl::host_accessor acc_other {*buf_other, sycl::write_only, sycl::no_init};

        const u32 nvar_loc = nvar;
        const u32 start_enque_loc = start_enque;


        for(u32 gid = 0; gid < sz; gid++){

            u32 _sz = sz;
        
            const u32 idx_extr = acc_idxs[gid] * nvar_loc;
            const u32 idx_push = start_enque_loc + gid * nvar_loc;
    
            for (u32 a = 0; a < nvar_loc; a++) {
                auto tmp = acc_curr[idx_extr + a];
                acc_other[idx_push + a] = tmp;
            }  

        }

        //printf("extracting : (%u)\n",sz);
        //for(u32 i = 0; i < sz; i++){
        //    printf("%u ",acc_idxs[i]);
        //}
        //printf("\n");
    }
    #endif
    
    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

        sycl::accessor acc_curr {*buf, cgh, sycl::read_only};
        sycl::accessor acc_other {*buf_other, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor acc_idxs {idxs_buf, cgh, sycl::read_only};

        const u32 nvar_loc = nvar;
        const u32 start_enque_loc = start_enque;

        cgh.parallel_for(sycl::range<1>{sz},  [=](sycl::item<1> i){

            const u32 gid = i.get_linear_id();
        
            const u32 idx_extr = acc_idxs[gid] * nvar_loc;
            const u32 idx_push = start_enque_loc + gid * nvar_loc;
    
            for (u32 a = 0; a < nvar_loc; a++) {
                acc_other[idx_push + a] = acc_curr[idx_extr + a];
            }       

        });

    });

    

}

template <class T> void PatchDataField<T>::append_subset_to(const std::vector<u32> &idxs, PatchDataField &pfield) const {

    if (pfield.nvar != nvar)
        throw shamrock_exc("field must be similar for extraction");

    const u32 start_enque = pfield.val_cnt;

    const u32 nvar = get_nvar();

    //pfield.expand(idxs.size());

    



    using buf_t = std::unique_ptr<sycl::buffer<T>>;

    const buf_t & buf_other  = pfield.get_buf();

    u32 sz = idxs.size();


    

    sycl::buffer<u32> idxs_buf(idxs.data(), sz);

    append_subset_to(idxs_buf,sz,pfield);
    

}



template<class T> class PdatField_insert_element;

template<class T> void PatchDataField<T>::insert_element(T v){
    u32 ins_pos = val_cnt;
    expand(1);

    shamsys::instance::get_compute_queue().submit([&] (sycl::handler& cgh) {

        auto id_ins = ins_pos;
        auto val = v;
        sycl::accessor acc {*buf, cgh, sycl::write_only};


        cgh.single_task<PdatField_insert_element<T>>([=] () {
            acc[id_ins] = val;
        });
    });

}








template<class T> class PdatField_apply_offset;

template<class T> void PatchDataField<T>::apply_offset(T off){

    shamsys::instance::get_compute_queue().submit([&] (sycl::handler& cgh) {
        
        auto val = off;
        sycl::accessor acc {*buf, cgh, sycl::read_write};

        cgh.parallel_for<PdatField_apply_offset<T>>(sycl::range<1>{val_cnt}, [=](sycl::id<1> idx){
            acc[idx] += val;
        });
    });

}



template<class T> class PdatField_insert;

template<class T> void PatchDataField<T>::insert(PatchDataField<T> &f2){

    const u32 old_val_cnt = val_cnt;//field_data.size();
    expand(f2.obj_cnt);

    shamsys::instance::get_compute_queue().submit([&] (sycl::handler& cgh) {
        
        const u32 idx_st = old_val_cnt;
        sycl::accessor acc {*buf, cgh};
        sycl::accessor acc_f2 {*f2.get_buf(), cgh};

        cgh.parallel_for<PdatField_insert<T>>(sycl::range<1>{f2.val_cnt}, [=](sycl::id<1> idx){
            acc[idx_st + idx] = acc_f2[idx];
        });
    });



}
















































//////////////////////////////////////////////////////////////////////////
// Define the patchdata field for all classes in XMAC_LIST_ENABLED_FIELD
//////////////////////////////////////////////////////////////////////////

#define X(a) template class PatchDataField<a>;
XMAC_LIST_ENABLED_FIELD
#undef X

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////
// data mocking for patchdata field
//////////////////////////////////////////////////////////////////////////

const u32 obj_mock_cnt = 6000;

template <> void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

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
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_2{distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_3{distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_8{distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                           distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f32_16{distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64(distf64(eng));
        }
    }
}

template <> void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_2{distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_3{distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_4{distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_8{distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                           distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = f64_16{distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng), distf64(eng),
                            distf64(eng), distf64(eng), distf64(eng), distf64(eng)};
        }
    }
}

template <> void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = distu32(eng);
        }
    }
}
template <> void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = distu64(eng);
        }
    }
}

template <> void PatchDataField<u32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = u32_3{distu32(eng),distu32(eng),distu32(eng)};
        }
    }
}
template <> void PatchDataField<u64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng) {
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1, obj_mock_cnt);

    {
        auto & buf = get_buf();
        sycl::host_accessor acc{*buf, sycl::write_only, sycl::no_init};

        for (u32 i = 0; i < val_cnt; i++) {
            acc[i] = u64_3{distu64(eng),distu64(eng),distu64(eng)};
        }
    }
}