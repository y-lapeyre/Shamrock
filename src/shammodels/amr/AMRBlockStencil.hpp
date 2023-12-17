// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRBlock.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief utility to manipulate AMR blocks
 */


#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include <array>
#include <variant>
namespace shammodels::amr::block {


    struct SameLevel{
        u32 block_idx;
    };

    struct Levelp1{
        std::array<u32,8> block_child_idxs;
    };

    struct Levelm1{

        /**
        * @brief Represent the position relative to wanted cell
        * format : (xyz)
        * exemple in 2D 
        * _______________________
        * |          |          |
        * |    mM    |    MM    |
        * |          |          |
        * -----------------------
        * |          |          |
        * |    mm    |    Mm    |
        * |          |          |
        * _______________________
        */
        enum STATE{
            mmm = 0, 
            mmM = 1, 
            mMm = 2, 
            mMM = 3,
            Mmm = 4, 
            MmM = 5, 
            MMm = 6, 
            MMM = 7, 
        };
        STATE neighbourgh_state;
        u32 block_idx;
    };



    /**
    * @brief Stencil element, describe the state of a cell relative to another
    * 
    */
    struct alignas(8) StencilElement{

        struct None{};
        
        std::variant<SameLevel, Levelm1, Levelp1, None> _int = None{};

        explicit StencilElement(SameLevel st) :_int(st){}
        explicit StencilElement(Levelm1 st) :_int(st){}
        explicit StencilElement(Levelp1 st) :_int(st){}
        StencilElement() = default;

        template<class Visitor1,class Visitor2,class Visitor3,class Visitor4>
        inline void visitor(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3, Visitor4 && f4){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameLevel>){
                    f1(arg);
                }else if constexpr (std::is_same_v<T, Levelm1>){
                    f2(arg);
                }else if constexpr (std::is_same_v<T, Levelp1>){
                    f3(arg);
                }else if constexpr (std::is_same_v<T, None>){
                    f4(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }

        template<class Tret,class Visitor1,class Visitor2,class Visitor3,class Visitor4>
        inline Tret visitor_ret(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3, Visitor4 && f4){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameLevel>){
                    return f1(arg);
                }else if constexpr (std::is_same_v<T, Levelm1>){
                    return f2(arg);
                }else if constexpr (std::is_same_v<T, Levelp1>){
                    return f3(arg);
                }else if constexpr (std::is_same_v<T, None>){
                    return f4(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }
    };  

} // namespace shammodels::amr::block