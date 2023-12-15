// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file StencilGenerator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
#include "shammodels/amr/basegodunov/AMRBlockStencil.hpp"


namespace shammodels::basegodunov::modules {

    struct StencilCache{

        static constexpr u32 dim = 3;
        using AMRBlock = amr::AMRBlock<f64_3, i64_3, 2>;

        enum BlockStencilElementId{
            block_xp = 0,
            block_xm = 1,
            block_yp = 2,
            block_ym = 3,
            block_zp = 4,
            block_zm = 5,
        };

        enum CellStencilElementId{
            cell_xp = 0,
            cell_xm = 1,
            cell_yp = 2,
            cell_ym = 3,
            cell_zp = 4,
            cell_zm = 5,
        };

        template<CellStencilElementId cell_st_dir>
        inline static constexpr BlockStencilElementId get_block_stencil_el(){
            if constexpr(cell_st_dir == cell_xp){
                return block_xp;
            }else 
            if constexpr(cell_st_dir == cell_xm){
                return block_xm;
            }else 
            if constexpr(cell_st_dir == cell_yp){
                return block_yp;
            }else 
            if constexpr(cell_st_dir == cell_ym){
                return block_ym;
            }else 
            if constexpr(cell_st_dir == cell_zp){
                return block_zp;
            }else 
            if constexpr(cell_st_dir == cell_zm){
                return block_zm;
            }else { 
                static_assert(shambase::always_false_v<decltype(cell_st_dir)>, "non-exhaustive visitor!");
            }
        }





        template<CellStencilElementId cell_st_dir>
        inline constexpr StencilElement lower_block_to_cell(
            u32 block_id,
            u32 local_block_index,
            std::array<u32, dim> coord,
            StencilElement block_stencil_el)
        {
            constexpr BlockStencilElementId block_st_dir = get_block_stencil_el<cell_st_dir>();

            auto get_cell_offset = [](){
                if constexpr(cell_st_dir == cell_xp){
                    return std::array<i32,3>{1,0,0};
                }else 
                if constexpr(cell_st_dir == cell_xm){
                    return std::array<i32,3>{-1,0,0};
                }else 
                if constexpr(cell_st_dir == cell_yp){
                    return std::array<i32,3>{0,1,0};
                }else 
                if constexpr(cell_st_dir == cell_ym){
                    return std::array<i32,3>{0,-1,0};
                }else 
                if constexpr(cell_st_dir == cell_zp){
                    return std::array<i32,3>{0,0,1};
                }else 
                if constexpr(cell_st_dir == cell_zm){
                    return std::array<i32,3>{0,0,-1};
                }else { 
                    static_assert(shambase::always_false_v<decltype(cell_st_dir)>, "non-exhaustive visitor!");
                }
            };

            auto get_cell_idx = [](u32 block_id, std::array<u32, 3> lcoord){
                return block_id*AMRBlock::block_size + AMRBlock::get_index(lcoord);
            };


            auto make_array_signed = [](std::array<u32, 3> arr) -> std::array<i32, 3>{
                return {static_cast<i32>(arr[0]),static_cast<i32>(arr[1]),static_cast<i32>(arr[2])};
            };

            auto make_array_unsigned = [](std::array<i32, 3> arr) -> std::array<u32, 3>{
                return {static_cast<u32>(arr[0]),static_cast<u32>(arr[1]),static_cast<u32>(arr[2])};
            };

            constexpr std::array<i32,3> off = get_cell_offset();

            //offset local coordinates
            std::array<i32, dim> coord_off = {
                coord[0] + off[0],
                coord[1] + off[1],
                coord[2] + off[2],
            };

            //check if still within block (and return if thats the case)
            auto early_ret_cell_idx = StencilElement(SameLevel{get_cell_idx(block_id,make_array_unsigned(coord_off))});

            if constexpr(cell_st_dir == cell_xp){
                if(coord_off[0] < AMRBlock::side_size){
                    return early_ret_cell_idx;
                }
            }else if constexpr(cell_st_dir == cell_xm){
                if(coord_off[0] > 0){
                    return early_ret_cell_idx;
                }
            }else if constexpr(cell_st_dir == cell_yp){
                if(coord[1] < AMRBlock::side_size){
                    return early_ret_cell_idx;
                }
            }else if constexpr(cell_st_dir == cell_ym){
                if(coord[1] > 0){
                    return early_ret_cell_idx;
                }
            }else if constexpr(cell_st_dir == cell_zp){
                if(coord[2] < AMRBlock::side_size){
                    return early_ret_cell_idx;
                }
            }else if constexpr(cell_st_dir == cell_zm){
                if(coord[2] > 0){
                    return early_ret_cell_idx;
                }
            }else { 
                static_assert(shambase::always_false_v<decltype(cell_st_dir)>, "non-exhaustive visitor!");
            }


            // the previous part has not returned so we have to look at the neighbouring block

            //make the local coord relative to the neighbouring block
            if constexpr(cell_st_dir == cell_xp){
                coord_off[0] -= AMRBlock::side_size;
            }else if constexpr(cell_st_dir == cell_xm){
                coord_off[0] += AMRBlock::side_size;
            }else if constexpr(cell_st_dir == cell_yp){
                coord_off[1] -= AMRBlock::side_size;
            }else if constexpr(cell_st_dir == cell_ym){
                coord_off[1] += AMRBlock::side_size;
            }else if constexpr(cell_st_dir == cell_zp){
                coord_off[2] -= AMRBlock::side_size;
            }else if constexpr(cell_st_dir == cell_zm){
                coord_off[2] += AMRBlock::side_size;
            }else { 
                static_assert(shambase::always_false_v<decltype(cell_st_dir)>, "non-exhaustive visitor!");
            }

            return block_stencil_el.visitor_ret<StencilElement>(
                [&](SameLevel && st){
                    return get_cell_idx(st.obj_idx,make_array_unsigned(coord_off));
                }, 
                [&](Levelm1 && st){

                    std::array<u32, 3> mod_coord {
                        u32(coord_off[0]) %2,
                        u32(coord_off[1]) %2,
                        u32(coord_off[2]) %2,
                    };

                    std::array<i32, 3> block_pos_offset{
                        (0b100 & st.neighbourgh_state) >> 2,
                        (0b010 & st.neighbourgh_state) >> 1,
                        (0b001 & st.neighbourgh_state) >> 0,
                    };

                    block_pos_offset[0] *= AMRBlock::side_size/2;
                    block_pos_offset[1] *= AMRBlock::side_size/2;
                    block_pos_offset[2] *= AMRBlock::side_size/2;

                    coord_off = {
                        block_pos_offset[0] + (coord_off[0]/2),
                        block_pos_offset[1] + (coord_off[1]/2),
                        block_pos_offset[2] + (coord_off[2]/2),
                    };

                    u32 neigh_state = mod_coord[0]*4 + mod_coord[1]*2 + mod_coord[2];

                    return StencilElement(Levelm1{
                            neigh_state,
                            get_cell_idx(st.obj_idx,make_array_unsigned(coord_off))
                        });
                }, 
                [&](Levelp1 && st){
                    
                    std::array<u32, 3> mod_coord {
                        u32(coord_off[0] / (AMRBlock::side_size/2)) ,
                        u32(coord_off[1] / (AMRBlock::side_size/2)) ,
                        u32(coord_off[2] / (AMRBlock::side_size/2)) ,
                    };
                    u32 child_select = mod_coord[0]*4 + mod_coord[1]*2 + mod_coord[2];

                    coord_off = {
                        int((coord_off[0] * 2) % AMRBlock::side_size),
                        int((coord_off[1] * 2) % AMRBlock::side_size),
                        int((coord_off[2] * 2) % AMRBlock::side_size)
                    };

                    //u32 off = st.obj_child_idxs[child_select]*AMRBlock::block_size;


                    //return StencilElement(Levelp1{{
                    //        off + AMRBlock::get_index({u32(coord_off[0])+0,u32(coord_off[1])+0,u32(coord_off[2])+0}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+0,u32(coord_off[1])+0,u32(coord_off[2])+1}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+0,u32(coord_off[1])+1,u32(coord_off[2])+0}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+0,u32(coord_off[1])+1,u32(coord_off[2])+1}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+1,u32(coord_off[1])+0,u32(coord_off[2])+0}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+1,u32(coord_off[1])+0,u32(coord_off[2])+1}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+1,u32(coord_off[1])+1,u32(coord_off[2])+0}),
                    //        off + AMRBlock::get_index({u32(coord_off[0])+1,u32(coord_off[1])+1,u32(coord_off[2])+1}),
                    //    }});

                    // is equivalent to

                    //return StencilElement(Levelp1{std::array<u32, 8>{
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +0 +Nside *0 +Nside * Nside *0,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +0 +Nside *0 +Nside * Nside *1,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +0 +Nside *1 +Nside * Nside *0,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +0 +Nside *1 +Nside * Nside *1,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +1 +Nside *0 +Nside * Nside *0,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +1 +Nside *0 +Nside * Nside *1,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +1 +Nside *1 +Nside * Nside *0,
                    //        off + (coord_off[0]) + Nside *coord_off[1]+ Nside * Nside *coord_off[2] +1 +Nside *1 +Nside * Nside *1,
                    //    }});

                    //is equivalent to

                    u32 idx_tmp = get_cell_idx(st.obj_child_idxs[child_select],make_array_unsigned(coord_off)); 

                    return StencilElement(Levelp1{{
                            idx_tmp + AMRBlock::get_index({0,0,0}),
                            idx_tmp + AMRBlock::get_index({0,0,1}),
                            idx_tmp + AMRBlock::get_index({0,1,0}),
                            idx_tmp + AMRBlock::get_index({0,1,1}),
                            idx_tmp + AMRBlock::get_index({1,0,0}),
                            idx_tmp + AMRBlock::get_index({1,0,1}),
                            idx_tmp + AMRBlock::get_index({1,1,0}),
                            idx_tmp + AMRBlock::get_index({1,1,1}),
                        }}); 
                });
        }


        template<CellStencilElementId cell_st_dir, BlockStencilElementId block_st_dir>
        inline constexpr StencilElement lower_block_to_cell(
            u32 block_id,
            u32 local_block_index,
            StencilElement block_stencil_el)
        {
            return lower_block_to_cell<cell_st_dir,block_st_dir>(block_id,local_block_index, AMRBlock::get_coord(local_block_index),
            block_stencil_el);
        }


    };

    template<class Tvec, class TgridVec>
    class StencilGenerator {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec,TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        using StencilElem = typename shammodels::basegodunov::StencilElement;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        StencilGenerator(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}



        private:
        void fill_slot(i64_3 relative_offset);

        
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::basegodunov::modules