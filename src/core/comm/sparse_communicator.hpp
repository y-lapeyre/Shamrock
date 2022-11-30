#pragma once

#include "aliases.hpp"
#include "core/patch/base/patch.hpp"
#include <vector>

#include "core/io/logs.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/sys/mpi_handler.hpp"



template <class T> using SparseCommSource = std::vector<std::unique_ptr<T>>;
template <class T> using SparseCommResult = std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<T>>>>;



class SparsePatchCommunicator;

template <class T> 
struct SparseCommExchanger{
    static SparseCommResult<T> sp_xchg(SparsePatchCommunicator & communicator, const SparseCommSource<T> &send_comm_pdat);
};

class SparsePatchCommunicator {
    

    std::vector<i32> local_comm_tag;

    std::vector<u64_2> global_comm_vec;
    std::vector<i32> global_comm_tag;

    u64 xcgh_byte_cnt = 0;


  public:
    std::vector<Patch> & global_patch_list;
    std::vector<u64_2> send_comm_vec;

    SparsePatchCommunicator(std::vector<Patch> & global_patch_list, std::vector<u64_2> send_comm_vec)
        : global_patch_list(global_patch_list), send_comm_vec(std::move(send_comm_vec)),
          local_comm_tag(send_comm_vec.size()) {}

    inline void fetch_comm_table() {

        {
            i32 iterator = 0;
            for (u64 i = 0; i < send_comm_vec.size(); i++) {
                local_comm_tag[i] = iterator;
                iterator++;
            }
        }

        auto timer_allgatherv = timings::start_timer("allgatherv", timings::mpi);
        mpi_handler::vector_allgatherv(send_comm_vec, mpi_type_u64_2, global_comm_vec, mpi_type_u64_2, MPI_COMM_WORLD);
        mpi_handler::vector_allgatherv(local_comm_tag, mpi_type_i32, global_comm_tag, mpi_type_i32, MPI_COMM_WORLD);
        timer_allgatherv.stop();

        xcgh_byte_cnt += 
            (send_comm_vec.size() * sizeof(u64) * 2)  + 
            (global_comm_vec.size() * sizeof(u64) * 2)  + 
            (local_comm_tag.size() * sizeof(i32))  + 
            (global_comm_tag.size() * sizeof(i32));
    }

    [[nodiscard]] inline u64 get_xchg_byte_count() const {
        return xcgh_byte_cnt;
    }

    inline void reset_xchg_byte_count(){
        xcgh_byte_cnt = 0;
    }

    template<typename T>
    friend class SparseCommExchanger;

    template<class T>
    inline SparseCommResult<T> sparse_exchange(const SparseCommSource<T> &send_comm_pdat){
        return SparseCommExchanger<T>::sp_xchg(*this, send_comm_pdat);
    }
};


