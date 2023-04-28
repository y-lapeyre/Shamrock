// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "shamrock/tree/TreeStructure.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/SyclHelper.hpp"
#include "shamsys/SyclMpiTypes.hpp"
#include "shamsys/comm/CommBuffer.hpp"
#include "shamsys/comm/CommRequests.hpp"
#include "shamsys/comm/ProtocolEnum.hpp"
#include "shambase/exception.hpp"
#include "shamsys/comm/details/CommImplBuffer.hpp"

#include <optional>
#include <stdexcept>

namespace shamsys::comm::details {

    template<class u_morton>
    class CommDetails<shamrock::tree::TreeStructure<u_morton>> {

        public:
        u32 internal_cell_count;
        bool one_cell_mode;

        inline CommDetails() = default;

        inline CommDetails(
            u32 internal_cell_count, bool one_cell_mode
        )
            : internal_cell_count(internal_cell_count), one_cell_mode(one_cell_mode){}

        inline CommDetails(shamrock::tree::TreeStructure<u_morton> &strc)
            : internal_cell_count(strc.internal_cell_count), one_cell_mode(strc.one_cell_mode) {}
    };


    template<class u_morton, Protocol comm_mode>
    class CommBuffer<shamrock::tree::TreeStructure<u_morton>, comm_mode> {

        using TreeStructure = shamrock::tree::TreeStructure<u_morton>;
        using CDetails = CommDetails<shamrock::tree::TreeStructure<u_morton>>;

        CDetails details;

        CommBuffer<sycl::buffer<u32>,comm_mode> buf_lchild_id  ; // size = internal
        CommBuffer<sycl::buffer<u32>,comm_mode> buf_rchild_id  ; // size = internal
        CommBuffer<sycl::buffer<u8>,comm_mode>  buf_lchild_flag; // size = internal
        CommBuffer<sycl::buffer<u8>,comm_mode>  buf_rchild_flag; // size = internal
        CommBuffer<sycl::buffer<u32>,comm_mode> buf_endrange   ; // size = internal (+1 if one cell mode)

        CommBuffer(
            
            CommBuffer<sycl::buffer<u32>,comm_mode>&& buf_lchild_id,  // size = internal
            CommBuffer<sycl::buffer<u32>,comm_mode> && buf_rchild_id,  // size = internal
            CommBuffer<sycl::buffer<u8>,comm_mode> && buf_lchild_flag, // size = internal
            CommBuffer<sycl::buffer<u8>,comm_mode> && buf_rchild_flag, // size = internal
            CommBuffer<sycl::buffer<u32>,comm_mode> && buf_endrange,
            CDetails && details
        )
            : 
            buf_lchild_id(std::move(buf_lchild_id)), 
            buf_rchild_id(std::move(buf_rchild_id)), 
            buf_lchild_flag(std::move(buf_lchild_flag)), 
            buf_rchild_flag(std::move(buf_rchild_flag)), 
            buf_endrange(std::move(buf_endrange)), 
            details(std::move(details)) {}

        public:

        inline CommBuffer(CDetails det) :
            details(det),
            buf_lchild_id{CommDetails<sycl::buffer<u32>>{}},
            buf_rchild_id{CommDetails<sycl::buffer<u32>>{}},
            buf_lchild_flag{CommDetails<sycl::buffer<u8>>{}},
            buf_rchild_flag{CommDetails<sycl::buffer<u8>>{}},
            buf_endrange{CommDetails<sycl::buffer<u32>>{}}
        {}

        inline CommBuffer(TreeStructure &obj_ref)
            : details(obj_ref),
              buf_lchild_id(*obj_ref.buf_lchild_id) ,
              buf_rchild_id(*obj_ref.buf_rchild_id) ,
              buf_lchild_flag(*obj_ref.buf_lchild_flag) ,
              buf_rchild_flag(*obj_ref.buf_rchild_flag) ,
              buf_endrange(*obj_ref.buf_endrange)
              {}

        inline CommBuffer(TreeStructure &obj_ref, CDetails details)
            : details(details),
              buf_lchild_id(*obj_ref.buf_lchild_id, details.internal_cell_count) ,
              buf_rchild_id(*obj_ref.buf_rchild_id, details.internal_cell_count) ,
              buf_lchild_flag(*obj_ref.buf_lchild_flag, details.internal_cell_count) ,
              buf_rchild_flag(*obj_ref.buf_rchild_flag, details.internal_cell_count) ,
              buf_endrange(*obj_ref.buf_endrange, details.internal_cell_count + (details.one_cell_mode ? 1 : 0))
              {}

        inline CommBuffer(TreeStructure && moved_obj) // TODO avoid using the copy constructor here
            : details(moved_obj),
              buf_lchild_id(*moved_obj.buf_lchild_id) ,
              buf_rchild_id(*moved_obj.buf_rchild_id) ,
              buf_lchild_flag(*moved_obj.buf_lchild_flag) ,
              buf_rchild_flag(*moved_obj.buf_rchild_flag) ,
              buf_endrange(*moved_obj.buf_endrange)
              {}

        inline CommBuffer(TreeStructure && moved_obj, CDetails details) // TODO avoid using the copy constructor here
            : details(details),
              buf_lchild_id(*moved_obj.buf_lchild_id, details.internal_cell_count) ,
              buf_rchild_id(*moved_obj.buf_rchild_id, details.internal_cell_count) ,
              buf_lchild_flag(*moved_obj.buf_lchild_flag, details.internal_cell_count) ,
              buf_rchild_flag(*moved_obj.buf_rchild_flag, details.internal_cell_count) ,
              buf_endrange(*moved_obj.buf_endrange, details.internal_cell_count + (details.one_cell_mode ? 1 : 0))
              {}

        inline TreeStructure copy_back(){
            auto lchild_id   = std::make_unique<sycl::buffer<u32>>(buf_lchild_id.copy_back()  );  
            auto rchild_id   = std::make_unique<sycl::buffer<u32>>(buf_rchild_id.copy_back()  );  
            auto lchild_flag = std::make_unique<sycl::buffer<u8> >(buf_lchild_flag.copy_back()); 
            auto rchild_flag = std::make_unique<sycl::buffer<u8> >(buf_rchild_flag.copy_back()); 
            auto endrange    = std::make_unique<sycl::buffer<u32>>(buf_endrange.copy_back()   );   

            return TreeStructure{details.internal_cell_count,details.one_cell_mode,
                std::move(lchild_id  ),
                std::move(rchild_id  ),
                std::move(lchild_flag),
                std::move(rchild_flag),
                std::move(endrange   )
            };
        }

        inline static TreeStructure convert(CommBuffer &&buf) {
            auto lchild_id   = std::make_unique<sycl::buffer<u32>>(buf.buf_lchild_id.copy_back()  );  
            auto rchild_id   = std::make_unique<sycl::buffer<u32>>(buf.buf_rchild_id.copy_back()  );  
            auto lchild_flag = std::make_unique<sycl::buffer<u8> >(buf.buf_lchild_flag.copy_back()); 
            auto rchild_flag = std::make_unique<sycl::buffer<u8> >(buf.buf_rchild_flag.copy_back()); 
            auto endrange    = std::make_unique<sycl::buffer<u32>>(buf.buf_endrange.copy_back()   );   

            return TreeStructure{buf.details.internal_cell_count,buf.details.one_cell_mode,
                std::move(lchild_id  ),
                std::move(rchild_id  ),
                std::move(lchild_flag),
                std::move(rchild_flag),
                std::move(endrange   )
            };
        }

        inline void isend(CommRequests &rqs, u32 rank_dest, u32 comm_flag, MPI_Comm comm) {
            buf_lchild_id  .isend(rqs, rank_dest, comm_flag, comm);
            buf_rchild_id  .isend(rqs, rank_dest, comm_flag, comm);
            buf_lchild_flag.isend(rqs, rank_dest, comm_flag, comm);
            buf_rchild_flag.isend(rqs, rank_dest, comm_flag, comm);
            buf_endrange   .isend(rqs, rank_dest, comm_flag, comm);
        }

        inline void irecv(CommRequests &rqs, u32 rank_src, u32 comm_flag, MPI_Comm comm) {
            buf_lchild_id  .irecv(rqs, rank_src, comm_flag, comm);
            buf_rchild_id  .irecv(rqs, rank_src, comm_flag, comm);
            buf_lchild_flag.irecv(rqs, rank_src, comm_flag, comm);
            buf_rchild_flag.irecv(rqs, rank_src, comm_flag, comm);
            buf_endrange   .irecv(rqs, rank_src, comm_flag, comm);
        }

        inline static CommBuffer irecv_probe(
            CommRequests &rqs,
            u32 rank_src,
            u32 comm_flag,
            MPI_Comm comm,
            CDetails details
        ) {

            auto tbuf_lchild_id   = CommBuffer<sycl::buffer<u32>, comm_mode>::irecv_probe(rqs, rank_src, comm_flag, comm, {});
            auto tbuf_rchild_id   = CommBuffer<sycl::buffer<u32>, comm_mode>::irecv_probe(rqs, rank_src, comm_flag, comm, {});
            auto tbuf_lchild_flag = CommBuffer<sycl::buffer<u8>, comm_mode>::irecv_probe(rqs, rank_src, comm_flag, comm, {});
            auto tbuf_rchild_flag = CommBuffer<sycl::buffer<u8>, comm_mode>::irecv_probe(rqs, rank_src, comm_flag, comm, {});
            auto tbuf_endrange    = CommBuffer<sycl::buffer<u32>, comm_mode>::irecv_probe(rqs, rank_src, comm_flag, comm, {});

            u64 cnt_recv1 = tbuf_lchild_id.get_details().comm_len;
            u64 cnt_recv2 = tbuf_endrange.get_details().comm_len;

            u32 internal_cell_count = cnt_recv1;
            bool one_cell_mode = cnt_recv1+1 == cnt_recv2;

            details.internal_cell_count = internal_cell_count;
            details.one_cell_mode = one_cell_mode;

            return CommBuffer{
                std::move(tbuf_lchild_id  ),
                std::move(tbuf_rchild_id  ),
                std::move(tbuf_lchild_flag),
                std::move(tbuf_rchild_flag),
                std::move(tbuf_endrange   ),
                std::move(details)
                };
        }

    };

}