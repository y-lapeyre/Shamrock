// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "patchdata_field.hpp"
#include "CL/sycl/usm.hpp"
#include "core/patch/base/enabled_fields.hpp"


template<class T>
void PatchDataField<T>::extract_element(u32 pidx, PatchDataField<T> & to){

    auto fast_extract_ptr = [](u32 idx, u32 lenght ,auto cnt){

        T end_ = cnt[lenght-1];
        T extr = cnt[idx];

        cnt[idx] = end_;

        return extr;
    };

    auto sub_extract = [fast_extract_ptr](u32 pidx, PatchDataField<T> & from, PatchDataField<T> & to){
        const u32 nvar = from.get_nvar();
        const u32 idx_val = pidx*nvar;
        const u32 idx_out_val = to.size();

        u32 from_sz = from.size();

        to.expand(1);

        {
            auto buf_to = to.data();
            auto buf_from = from.data();

            sycl::host_accessor acc_to{*buf_to};
            sycl::host_accessor acc_from{*buf_from};

            for(u32 i = nvar-1 ; i < nvar ; i--){
                acc_to[idx_out_val + i] = (fast_extract_ptr(idx_val + i,from_sz, acc_from));
            }
        }

        from.shrink(1);
    };

    sub_extract(pidx,*this,to);

}

template<class T>
bool PatchDataField<T>::check_field_match(PatchDataField<T> &f2){
    bool match = true;

    match = match && (field_name == f2.field_name);
    match = match && (nvar       == f2.nvar);
    match = match && (obj_cnt    == f2.obj_cnt);
    match = match && (val_cnt    == f2.val_cnt);

    //std::cout << "fieldname : " << field_name << std::endl;
    //std::cout << "val_cnt : " << val_cnt << std::endl;

    auto buf  = data();
    sycl::host_accessor acc{*buf};

    auto buf_f2  = f2.data();
    sycl::host_accessor acc_f2{*buf};

    for (u32 i = 0; i < val_cnt; i++) {
        //std::cout << i << " " << test_sycl_eq(data()[i],f2.data()[i]) << " " ;
        //print_vec(std::cout, data()[i]);
        //std::cout <<" ";
        //print_vec(std::cout, f2.data()[i]);
        //std::cout <<  std::endl;
        match = match && test_sycl_eq(acc[i],acc_f2[i]);
    }

    return match;
}


template<class T>
void PatchDataField<T>::append_subset_to(std::vector<u32> & idxs, PatchDataField & pfield){

    if(pfield.nvar != nvar) throw shamrock_exc("field must be similar for extraction");

    const u32 start_enque = pfield.val_cnt;

    const u32 nvar = get_nvar();

    pfield.expand(idxs.size());

    for (u32 i = 0; i < idxs.size(); i++) {

        const u32 idx_extr = idxs[i]*nvar;
        const u32 idx_push = start_enque + i*nvar;

        for(u32 a = 0; a < nvar ; a++){
            pfield.usm_data()[idx_push + a] = usm_data()[idx_extr + a];
        }

    }



    //auto buf_cur  = data();
    //auto buf_other  = data();
    //
    //{
    //    sycl::host_accessor acc_cur{*buf_cur};
    //    sycl::host_accessor acc_other{*buf_cur};
    //
    //    for (u32 i = 0; i < idxs.size(); i++) {
    //
    //        const u32 idx_extr = idxs[i]*nvar;
    //        const u32 idx_push = start_enque + i*nvar;
    //
    //        for(u32 a = 0; a < nvar ; a++){
    //            acc_other[idx_push + a] = acc_cur[idx_extr + a];
    //        }
    //
    //    }
    //}
    
}


















#define X(a) template class PatchDataField<a>;
XMAC_LIST_ENABLED_FIELD
#undef X





template<> void PatchDataField<f32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);

    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32(distf64(eng));
    }
    //for (auto & a : field_data) {
    //    a = f32(distf64(eng));
    //}
}

template<> void PatchDataField<f32_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32_2{distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f32_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f32_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}





template<> void PatchDataField<f64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64(distf64(eng));
    }
}

template<> void PatchDataField<f64_2>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64_2{distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_3>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64_3{distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_4>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64_4{distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_8>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64_8{distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)};
    }
}

template<> void PatchDataField<f64_16>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_real_distribution<f64> distf64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = f64_16{
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),
            distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng),distf64(eng)
            };
    }
}


template<> void PatchDataField<u32>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u32> distu32(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = distu32(eng);
    }
}
template<> void PatchDataField<u64>::gen_mock_data(u32 obj_cnt, std::mt19937 &eng){
    resize(obj_cnt);
    std::uniform_int_distribution<u64> distu64(1,6000);
    for (u32 i = 0; i < val_cnt; i++) {
        _data[i] = distu64(eng);
    }
}












namespace patchdata_field {

    comm_type current_mode = CopyToHost;


    template<class T>
    PatchDataFieldMpiRequest<T>::PatchDataFieldMpiRequest(
            PatchDataField<T> &pdat_field,
            comm_type comm_mode,
            op_type comm_op,
            u32 comm_sz
        ) : pdat_field(pdat_field), comm_sz(comm_sz), comm_op(comm_op), comm_mode(comm_mode){
            
        logger::debug_mpi_ln("PatchDataField MPI Comm", "starting mpi sycl comm ",comm_sz,comm_op,comm_mode);

        if(comm_mode == CopyToHost){



            if (comm_op == Isend) {


                comm_ptr = sycl::malloc_shared<T>(comm_sz, sycl_handler::get_compute_queue());
                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host",comm_sz,"->",comm_ptr);

                auto buf = pdat_field.data();

                if(pdat_field.size()> 0){
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                    auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){
                        sycl::accessor acc {*buf, cgh, sycl::read_only};

                        T* ptr = comm_ptr;

                        cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item){
                            ptr[item] = acc[item];
                        });

                    });

                    ker_copy.wait();
                }else{
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
                }

                

            }else if (comm_op == Irecv) {

                comm_ptr = sycl::malloc_shared<T>(comm_sz, sycl_handler::get_compute_queue());

                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host",comm_sz);

            }else{
                logger::err_ln("PatchDataField MPI Comm", "communication operation not implemented :",comm_op);
            }


        }else if(comm_mode == DirectGPU){
            
        

            if (comm_op == Isend) {


                comm_ptr = sycl::malloc_device<T>(comm_sz, sycl_handler::get_compute_queue());
                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host",comm_sz,"->",comm_ptr);

                auto buf = pdat_field.data();

                if(pdat_field.size()> 0){
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM");

                    auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){
                        sycl::accessor acc {*buf, cgh, sycl::read_only};

                        T* ptr = comm_ptr;

                        cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item){
                            ptr[item] = acc[item];
                        });

                    });

                    ker_copy.wait();
                }else{
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy buffer -> USM (skipped size=0)");
                }

                

            }else if (comm_op == Irecv) {

                comm_ptr = sycl::malloc_device<T>(comm_sz, sycl_handler::get_compute_queue());

                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::malloc_host",comm_sz);

            }else{
                logger::err_ln("PatchDataField MPI Comm", "communication operation not implemented :",comm_op);
            }


        }else{
            logger::err_ln("PatchDataField MPI Comm", "communication mode not implemented :",comm_mode);
        }

        

    }

    template<class T>
    void PatchDataFieldMpiRequest<T>::finalize(){

        logger::debug_mpi_ln("PatchDataField MPI Comm", "finalizing mpi sycl comm ",comm_sz,comm_op,comm_mode);


        pdat_field.resize(comm_sz);

        if(comm_mode == CopyToHost){
            if (comm_op == Isend) {

                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free",comm_ptr);

                sycl::free(comm_ptr, sycl_handler::get_compute_queue());

            }else if (comm_op == Irecv) {

                
                auto buf = pdat_field.data();

                if(pdat_field.size()> 0){
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                    auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){
                        sycl::accessor acc {*buf, cgh, sycl::write_only};

                        T* ptr = comm_ptr;

                        cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item){
                            acc[item] = ptr[item];
                        });

                    });

                    ker_copy.wait();
                }else{
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
                }



                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free",comm_ptr);

                sycl::free(comm_ptr, sycl_handler::get_compute_queue());

            }else{
                logger::err_ln("PatchDataField MPI Comm", "communication operation not implemented :",comm_op);
            }
        }else if(comm_mode == DirectGPU){
        
            if (comm_op == Isend) {

                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free",comm_ptr);

                sycl::free(comm_ptr, sycl_handler::get_compute_queue());

            }else if (comm_op == Irecv) {

                
                auto buf = pdat_field.data();

                if(pdat_field.size()> 0){
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer");

                    auto ker_copy = sycl_handler::get_compute_queue().submit([&](sycl::handler & cgh){
                        sycl::accessor acc {*buf, cgh, sycl::write_only};

                        T* ptr = comm_ptr;

                        cgh.parallel_for(sycl::range<1>{comm_sz}, [=](sycl::item<1> item){
                            acc[item] = ptr[item];
                        });

                    });

                    ker_copy.wait();
                }else{
                    logger::debug_sycl_ln("PatchDataField MPI Comm", "copy USM -> buffer (skipped size=0)");
                }



                logger::debug_sycl_ln("PatchDataField MPI Comm", "sycl::free",comm_ptr);

                sycl::free(comm_ptr, sycl_handler::get_compute_queue());

            }else{
                logger::err_ln("PatchDataField MPI Comm", "communication operation not implemented :",comm_op);
            }

        }else{
            logger::err_ln("PatchDataField MPI Comm", "communication mode not implemented :",comm_mode);
        }
        
    }


    #define X(a) template struct PatchDataFieldMpiRequest<a>;
    XMAC_LIST_ENABLED_FIELD
    #undef X
    



} // namespace patchdata_field

