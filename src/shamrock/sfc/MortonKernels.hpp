#pragma once

#include "aliases.hpp"
#include "shamrock/sfc/morton.hpp"




namespace shamrock::sfc{

    namespace details {
        /**
        * @brief fill the end of a buffer (indices from morton_count up to fill_count-1) with error values (maximum int value)
        * 
        * @param queue sycl queue
        * @param morton_count lenght of the morton buffer 
        * @param fill_count final lenght to be filled with error value
        * @param buf_morton morton buffer that will be updated
        */
        template<class morton_t, morton_t err_code>
        void sycl_fill_trailling_buffer(
            sycl::queue & queue,
            u32 morton_count,
            u32 fill_count,
            std::unique_ptr<sycl::buffer<morton_t>> & buf_morton
            );
    } // namespace details

    template<class morton_t,class pos_t, u32 dim>
    class MortonKernels{

        public:

        class xyz_to_Morton;
        
        using Morton = MortonCodes<morton_t, dim>;

        using ipos_t  = typename Morton::int_vec_repr;
        using int_t = typename Morton::int_vec_repr_base;

        

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
        static void sycl_xyz_to_morton(
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
        inline static void sycl_fill_trailling_buffer(
            sycl::queue & queue,
            u32 morton_count,
            u32 fill_count,
            std::unique_ptr<sycl::buffer<morton_t>> & buf_morton
            ){
                details::sycl_fill_trailling_buffer<morton_t, Morton::err_code>(queue, morton_count, fill_count, buf_morton);
        }


        template<class u_morton, class vec_prec>
        static void sycl_irange_to_range(sycl::queue & queue,
            u32 buf_len , 
            vec_prec bounding_box_min,
            vec_prec bounding_box_max,
            std::unique_ptr<sycl::buffer<ipos_t>> & buf_pos_min_cell,
            std::unique_ptr<sycl::buffer<ipos_t>> & buf_pos_max_cell,
            std::unique_ptr<sycl::buffer<vec_prec>> & out_buf_pos_min_cell_flt,
            std::unique_ptr<sycl::buffer<vec_prec>> & out_buf_pos_max_cell_flt);

    };

} // namespace shamrock::sfc