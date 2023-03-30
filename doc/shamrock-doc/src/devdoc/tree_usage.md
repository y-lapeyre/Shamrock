# Shamrock Radix Tree usage

## Information

## Setup 


```c++

RadixTree<morton_mode, vector_type, ..dimension..> rtree = 

    RadixTree<morton_mode, vector_type, ..dimension..>(
        shamsys::instance::get_compute_queue(),
        {coord_range.lower, coord_range.upper},
        pos,
        cnt,
        reduc_lev
    );

    rtree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
    rtree.convert_bounding_box(shamsys::instance::get_compute_queue());
    

```

## Tree traversal 


### Interaction Criterion 

```c++
template<class u_morton, class flt>
class SPHTestInteractionCrit {
    using vec = sycl::vec<flt, 3>;
    public:

    RadixTree<u_morton, vec, 3> &tree;
    sycl::buffer<vec> &positions;
    u32 part_count;
    flt Rpart;

    class Access {
        public:

        sycl::accessor<vec, 1, sycl::access::mode::read> part_pos;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_min;
        sycl::accessor<vec, 1, sycl::access::mode::read> tree_cell_coordrange_max;

        flt Rpart;
        flt Rpart_pow2;

        Access(SPHTestInteractionCrit crit, sycl::handler &cgh)
            : part_pos{crit.positions, cgh, sycl::read_only}, Rpart(crit.Rpart),Rpart_pow2(crit.Rpart*crit.Rpart),tree_cell_coordrange_min{*crit.tree.buf_pos_min_cell_flt, cgh, sycl::read_only},
                tree_cell_coordrange_max{
                    *crit.tree.buf_pos_max_cell_flt, cgh, sycl::read_only} {}

        class ObjectValues {
            public:
            vec xyz_a;
            ObjectValues(Access acc, u32 index)
                : xyz_a(acc.part_pos[index]) {}
        };

    };

    inline static bool
    criterion(u32 node_index, Access acc, typename Access::ObjectValues current_values) {
        vec cur_pos_min_cell_b = acc.tree_cell_coordrange_min[node_index];
        vec cur_pos_max_cell_b = acc.tree_cell_coordrange_max[node_index];

        vec box_int_sz = {acc.Rpart,acc.Rpart,acc.Rpart};

        return 
            BBAA::cella_neigh_b(
                current_values.xyz_a - box_int_sz, current_values.xyz_a + box_int_sz, 
                cur_pos_min_cell_b, cur_pos_max_cell_b) ||
            BBAA::cella_neigh_b(
                current_values.xyz_a, current_values.xyz_a,                   
                cur_pos_min_cell_b - box_int_sz, cur_pos_min_cell_b + box_int_sz);
    };
};
```

### Traversal 


```c++
using Criterion = SPHTestInteractionCrit<morton_mode,flt>;
using CriterionAcc = typename Criterion::Access;
using CriterionVal = typename CriterionAcc::ObjectValues; 

using namespace shamrock::tree;
TreeStructureWalker walk = generate_walk<Recompute>(
    rtree.tree_struct, len_pos, SPHTestInteractionCrit<morton_mode,flt>{rtree,*pos, len_pos, rpart}
);


q.submit([&](sycl::handler &cgh) {
    auto walker        = walk.get_access(cgh);
    auto leaf_iterator = rtree.get_leaf_access(cgh);

    sycl::accessor neigh_count {neighbours,cgh,sycl::write_only, sycl::no_init};
    

    cgh.parallel_for(walker.get_sycl_range(), [=](sycl::item<1> item) {
        u32 sum = 0;

        CriterionVal int_values{walker.criterion(), static_cast<u32>(item.get_linear_id())};

        walker.for_each_node(
            item,int_values,
            [&](u32 /*node_id*/, u32 leaf_iterator_id) {
                leaf_iterator.iter_object_in_leaf(
                    leaf_iterator_id, [&](u32 obj_id) { 

                        vec xyz_b = walker.criterion().part_pos[obj_id];
                        vec dxyz = xyz_b - int_values.xyz_a;
                        flt dot_ = sycl::dot(dxyz,dxyz);

                        if(dot_ < walker.criterion().Rpart_pow2){
                            sum += 1; 
                        }
                    }
                );
            },
            [&](u32 node_id) {}
        );

        neigh_count[item] = sum;
    });
    
});
```