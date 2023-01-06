/*
cf https://github.com/tdavidcl/Shamrock/issues/23 


we want to user side to look like this


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

Communicator<... type ...> comm {MPI_COMM_WORLD, protocol::DirectGPU};

auto tmp = comm.prepare_send_full(... obj to send ...);
auto tmp = comm.prepare_send(... obj to send ...,details<... type ...>{.....});

// ... do comm calls
comm.isend(tmp,0 ,0);
// ...

comm.sync(); // note : sync only sync sycl with MPI, ie can be nonblocking

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

*/

#include "MpiWrapper.hpp"
#include "legacy/log.hpp"

#include <variant>
#include <vector>


namespace shamsys::communicator {

    enum Protocol{
        /**
         * @brief copy data to the host and then perform the call
         */
        CopyToHost, 
        
        /**
         * @brief copy data straight from the GPU
         */
        DirectGPU, 
        
        /**
         * @brief  copy data straight from the GPU & flatten sycl vector to plain arrays
         */
        DirectGPUFlatten,
    };

    template<class T> class Communicator;

} // namespace shamsys::communicator


namespace shamsys::communicator::details {

    enum CommOp{
        Send,Recv
    };

    /**
    * Requirement : constructor = create MPI pointer | destructor = wait + free mpi ptr
    */
    template<class T, CommOp op, Protocol mode> 
    class CommRequestBase; //implement the protocol for each cases





    /*
    currently missing : 
    - Way to specify details to the protocol (ex : only those field in patchdata)
    - way to wait call SYCL to check is MPI ready
    */






    template<class T, Protocol mode> 
    class CommRequestList{


        template<CommOp op>
        using Request = CommRequestBase<T,op,mode>;

        using var_t = std::variant<
                            Request<Send>,
                            Request<Recv>
                        >;

        std::vector<var_t> rqs;



        

        template<CommOp op> void add_rq(Request<op> && rq){
            rqs.push_back(rq);
        }

        inline void sync(){//must be done with variant too
            if(! rqs.empty()){
                std::vector<MPI_Request> addrs;

                for(auto a : rqs){
                    addrs.push_back(a.mpi_rq);
                }

                std::vector<MPI_Status> st_lst(addrs.size());
                mpi::waitall(addrs.size(), addrs.data(), st_lst.data());

                for(auto a : rqs){
                    a.sync();
                }

                rqs.clear();
            }
        }

        inline ~CommRequestList(){
            if(! rqs.empty()){
                sync();
            }else{
                logger::debug_ln("CommRequestList", "Warning sync() called implicitly");
            }
        }

    };


} // namespace shamsys::communicator::details