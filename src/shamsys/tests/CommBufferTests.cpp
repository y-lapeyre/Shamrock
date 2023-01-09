#include "shamtest/shamtest.hpp"
#include <random>

#include "shamsys/Comm.hpp"
#include "shamsys/CommImplBuffer.hpp"

TestStart(Unittest, "shamsys/comm/mpi-sycl/comm-buffer", commtestbuffer, 2){

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f32> distval(-1.0F, 1.0F);

    u32 npart = 1e6;

    sycl::buffer<f32_3> buf_comp (npart);


    {
        sycl::host_accessor acc {buf_comp};
        for(u32 i = 0; i < npart; i++){
            acc[i] = f32_3(distval(eng),distval(eng),distval(eng));
        }
    }


    using namespace shamsys::comm;

    CommBuffer buf {buf_comp, DirectGPU};
    
}