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








        inline constexpr StencilElement lower_block_to_cell_xp_xp(
            u32 block_id,
            u32 local_block_index,
            std::array<u32, dim> coord,
            StencilElement block_stencil_el)
        {
            if(coord[0] < AMRBlock::block_size-1){
            return StencilElement(SameLevel{
                    block_id*AMRBlock::block_size + AMRBlock::get_index(
                        {coord[0]+1, coord[1], coord[2]}
                    )
                });
            } 
            
            // wanted dir is in another block
            return block_stencil_el.visitor_ret<StencilElement>(
                [&](SameLevel sl){
                    return StencilElement{SameLevel{
                        sl.obj_idx*AMRBlock::block_size + AMRBlock::get_index(
                            {0, coord[1], coord[2]}
                        )
                    }};
                }, 
                [](Levelp1 sl){
                    return StencilElement{};
                }, 
                [](Levelm1 sl){
                    return StencilElement{};
                }
            );
                
        }























        template<CellStencilElementId cell_st_dir, BlockStencilElementId block_st_dir>
        inline constexpr StencilElement lower_block_to_cell(
            u32 block_id,
            u32 local_block_index,
            std::array<u32, dim> coord,
            StencilElement block_stencil_el)
        {

            if constexpr(cell_st_dir == cell_xp && block_st_dir == block_xp){
                return lower_block_to_cell_xp_xp(block_id, local_block_index, coord, block_stencil_el);
            }else if constexpr(cell_st_dir == cell_xm && block_st_dir == block_xm){
                if(coord[0] > 0){
                    return StencilElement(SameLevel{
                        block_id + AMRBlock::get_index(
                            {coord[0]-1, coord[1], coord[2]}
                        )
                    });
                }else{ // wanted dir is in another block

                }
            }else if constexpr(cell_st_dir == cell_yp && block_st_dir == block_yp){
                if(coord[1] < AMRBlock::block_size-1){
                    return StencilElement(SameLevel{
                        block_id + AMRBlock::get_index(
                            {coord[0], coord[1]+1, coord[2]}
                        )
                    });
                }else{ // wanted dir is in another block

                }
            }else if constexpr(cell_st_dir == cell_ym && block_st_dir == block_ym){
                if(coord[1] > 0){
                    return StencilElement(SameLevel{
                        block_id + AMRBlock::get_index(
                            {coord[0], coord[1]-1, coord[2]}
                        )
                    });
                }else{ // wanted dir is in another block

                }
            }else if constexpr(cell_st_dir == cell_zp && block_st_dir == block_zp){
                if(coord[2] < AMRBlock::block_size-1){
                    return StencilElement(SameLevel{
                        block_id + AMRBlock::get_index(
                            {coord[0], coord[1], coord[2]+1}
                        )
                    });
                }else{ // wanted dir is in another block

                }
            }else if constexpr(cell_st_dir == cell_zm && block_st_dir == block_zm){
                if(coord[2] > 0){
                    return StencilElement(SameLevel{
                        block_id + AMRBlock::get_index(
                            {coord[0], coord[1], coord[2]-1}
                        )
                    });
                }else{ // wanted dir is in another block

                }
            }else { 
                static_assert(shambase::always_false_v<decltype(cell_st_dir)>, "non-exhaustive visitor!");
            }


            return StencilElement{};
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