// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"

namespace shamalgs::memory {


    /**
     * @brief extract a value of a buffer 
     * 
     * @tparam T the type of the buffer & value
     * @param q the queue to use
     * @param buf the buffer to extract from
     * @param idx the index of the value that will be extracted
     * @return T the extracted value
     */
    template<class T> T extract_element(sycl::queue &q, sycl::buffer<T> &buf, u32 idx);

    template<class T> sycl::buffer<T> vec_to_buf(const std::vector<T> &buf);
    template<class T> std::vector<T> buf_to_vec(sycl::buffer<T> &buf, u32 len);


    /**
     * @brief enqueue a do nothing kernel to force the buffer to move
     * 
     * @tparam T 
     * @param q 
     * @param buf 
     */
    template<class T> inline void move_buffer_on_queue(sycl::queue & q, sycl::buffer<T> & buf){
        sycl::buffer<T> tmp(1);
        q.submit([&](sycl::handler & cgh){
            sycl::accessor a {buf, cgh, sycl::read_write};
            sycl::accessor b {tmp, cgh, sycl::write_only, sycl::no_init};

            cgh.single_task([=](){
                b[0] = a[0];
            });
        });
    }



    template<class T> inline void buf_fill(sycl::queue & q,sycl::buffer<T> & buf, T value){
        q.submit([=, &buf](sycl::handler & cgh){
            sycl::accessor acc {buf, cgh, sycl::write_only};
            cgh.fill(acc, value);
        });
    }

    template<class T> inline void buf_fill_discard(sycl::queue & q,sycl::buffer<T> & buf, T value){
        q.submit([=, &buf](sycl::handler & cgh){
            sycl::accessor acc {buf, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(acc, value);
        });
    }


    template<class T,typename... Tformat> 
    inline void print_buf(
        sycl::buffer<T> & buf,
        u32 len, 
        u32 column_count,
        fmt::format_string<Tformat...> fmt
    ){

        sycl::host_accessor acc {buf, sycl::read_only};

        std::string accum;

        for(u32 i = 0; i < len; i++){

            if(i%column_count == 0){
                if(i == 0){
                    accum += shambase::format("{:8} : ", i);
                }else{
                    accum += shambase::format("\n{:8} : ", i);
                }
            }

            accum += shambase::format(fmt, acc[i]);

        }

        logger::raw_ln(accum);

    }

    template<class T>
    void copybuf_discard(sycl::buffer<T> & source, sycl::buffer<T> & dest, u32 cnt){
        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor src {source,cgh,sycl::read_only};
            sycl::accessor dst {dest,cgh,sycl::write_only,sycl::no_init};

            cgh.parallel_for(sycl::range<1>{cnt},[=](sycl::item<1> i){
                dst[i] = src[i];
            });

        });
    }

    template<class T> 
    std::unique_ptr<sycl::buffer<T>> duplicate(const std::unique_ptr<sycl::buffer<T>> & buf_in){
        if(buf_in){
            auto buf = std::make_unique<sycl::buffer<T>>(buf_in->size());
            copybuf_discard(*buf_in,*buf, buf_in->size());
            return std::move(buf);
        }
        return {};
    }





}