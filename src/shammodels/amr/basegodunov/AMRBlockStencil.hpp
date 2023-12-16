#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include <array>
#include <variant>

namespace shammodels::basegodunov {
        
    struct SameBlockLevel{
        u32 block_idx;
    };

    struct BlockLevelp1{
        std::array<u32,8> block_child_idxs;
    };

    struct BlockLevelm1{

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
    struct alignas(8) BlockStencilElement{
        
        std::variant<SameBlockLevel, BlockLevelm1, BlockLevelp1> _int;

        explicit BlockStencilElement(SameBlockLevel st) :_int(st){}
        explicit BlockStencilElement(BlockLevelm1 st) :_int(st){}
        explicit BlockStencilElement(BlockLevelp1 st) :_int(st){}
        BlockStencilElement() = default;

        template<class Visitor1,class Visitor2,class Visitor3>
        inline void visitor(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameBlockLevel>){
                    f1(arg);
                }else if constexpr (std::is_same_v<T, BlockLevelm1>){
                    f2(arg);
                }else if constexpr (std::is_same_v<T, BlockLevelp1>){
                    f3(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }

        template<class Tret,class Visitor1,class Visitor2,class Visitor3>
        inline Tret visitor_ret(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameBlockLevel>){
                    return f1(arg);
                }else if constexpr (std::is_same_v<T, BlockLevelm1>){
                    return f2(arg);
                }else if constexpr (std::is_same_v<T, BlockLevelp1>){
                    return f3(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }
    };  










      
    struct SameCellLevel{
        u32 cell_idx;
    };

    struct BlockCellLevelp1{
        u32 cell_idx_mmm;
    };

    struct BlockCellLevelm1{

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
        u32 cell_idx;
    };





    /**
    * @brief Stencil element, describe the state of a cell relative to another
    * 
    */
    struct alignas(8) CellStencilElement{
        
        std::variant<SameCellLevel, BlockCellLevelm1, BlockCellLevelp1> _int;

        explicit CellStencilElement(SameCellLevel st) :_int(st){}
        explicit CellStencilElement(BlockCellLevelm1 st) :_int(st){}
        explicit CellStencilElement(BlockCellLevelp1 st) :_int(st){}
        CellStencilElement() = default;

        template<class Visitor1,class Visitor2,class Visitor3>
        inline void visitor(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameCellLevel>){
                    f1(arg);
                }else if constexpr (std::is_same_v<T, BlockCellLevelm1>){
                    f2(arg);
                }else if constexpr (std::is_same_v<T, BlockCellLevelp1>){
                    f3(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }

        template<class Tret,class Visitor1,class Visitor2,class Visitor3>
        inline Tret visitor_ret(Visitor1 && f1, Visitor2 && f2, Visitor3 && f3){
            std::visit([&](auto&& arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SameCellLevel>){
                    return f1(arg);
                }else if constexpr (std::is_same_v<T, BlockCellLevelm1>){
                    return f2(arg);
                }else if constexpr (std::is_same_v<T, BlockCellLevelp1>){
                    return f3(arg);
                }else { 
                    static_assert(shambase::always_false_v<T>, "non-exhaustive visitor!");
                }
            }, _int);
        }
    };  


}