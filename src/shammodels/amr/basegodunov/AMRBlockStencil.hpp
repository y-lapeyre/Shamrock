#include <array>
#include <variant>

struct SameLevel{
    int block_idx;
};

struct Levelp1{
    std::array<int,8> block_child_idxs;
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
    int block_idx;
};

/**
 * @brief Stencil element, describe the state of a cell relative to another
 * 
 */
struct alignas(8) StencilElement{
    
    std::variant<SameLevel, Levelm1, Levelp1> _int;

    

};  