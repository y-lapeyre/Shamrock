/**
 * @file main_sph.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-08-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "sys/sycl_handler.hpp"
#include "sys/mpi_handler.hpp"




/**
* \brief  Main function for the SPH side of the code
*/
int main(void){
	
    mpi_handler::init();


    init_sycl();

    mpi_handler::close();

}
