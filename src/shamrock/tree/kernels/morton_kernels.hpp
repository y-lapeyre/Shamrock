// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "aliases.hpp"
#include <memory>


/**
 * @brief convert a buffer of 3d positions to morton codes
 * 
 * @param queue sycl queue
 * @param pos_count lenght of the position buffer 
 * @param in_positions 
 * @param bounding_box_min 
 * @param bounding_box_max 
 * @param out_morton 
 */
template<class morton_t,class pos_t, u32 dim>
void sycl_xyz_to_morton(
    sycl::queue & queue,
    u32 pos_count,
    std::unique_ptr<sycl::buffer<pos_t>> & in_positions,
    pos_t bounding_box_min,
    pos_t bounding_box_max,
    std::unique_ptr<sycl::buffer<morton_t>> & out_morton);



/**
 * @brief fill the end of a buffer (indices from morton_count up to fill_count-1) with error values (maximum int value)
 * 
 * @param queue sycl queue
 * @param morton_count lenght of the morton buffer 
 * @param fill_count final lenght to be filled with error value
 * @param buf_morton morton buffer that will be updated
 */
template<class morton_t>
void sycl_fill_trailling_buffer(
    sycl::queue & queue,
    u32 morton_count,
    u32 fill_count,
    std::unique_ptr<sycl::buffer<morton_t>> & buf_morton
    );
