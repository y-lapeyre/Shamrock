/**
 * @file main_amr.cpp
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
* \brief  Main function for the AMR side of the code
*/
int main(void){
	
    mpi_init();


    init_sycl();

    mpi_close();

}
