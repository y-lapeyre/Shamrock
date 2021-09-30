#include "../../aliases.hpp"
#include "../../flags.hpp"


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
void sycl_xyz_to_morton(
    sycl::queue* queue,
    u32 pos_count,
    sycl::buffer<f3_d>* in_positions,
    f3_d bounding_box_min,
    f3_d bounding_box_max,
    sycl::buffer<u_morton>* out_morton);



/**
 * @brief fill the end of a buffer (indices from morton_count up to fill_count-1) with error values (maximum int value)
 * 
 * @param queue sycl queue
 * @param morton_count lenght of the morton buffer 
 * @param fill_count final lenght to be filled with error value
 * @param buf_morton morton buffer that will be updated
 */
void sycl_fill_trailling_buffer(
    sycl::queue* queue,
    u32 morton_count,
    u32 fill_count,
    sycl::buffer<u_morton>* buf_morton
    );