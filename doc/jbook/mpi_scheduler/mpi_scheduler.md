# SchedulerMPI

The MPI Scheduler is the class that handle the distribution of Patches / Data on the cluster.

## Scheduler step 


```{prf:algorithm} mpi scheduler step
1. Initialisation  
    &ensp;&thinsp; update patch list  
    &ensp;&thinsp; rebuild patch index map  
    &ensp;&thinsp; apply 2 reduction step patchtree  
    &ensp;&thinsp; Generate merge and split request  

2. Patch Splitting  
    &ensp;&thinsp; apply split requests  
    &ensp;&thinsp; update ```PatchTree```

3. Patch Merging & LB  
    &ensp;&thinsp; update packing index  
    &ensp;&thinsp; update patch list    
    &ensp;&thinsp; generate LB change list  
    &ensp;&thinsp; apply LB change list  
    &ensp;&thinsp; apply merge requests    
    &ensp;&thinsp; update ```PatchTree```  
    &ensp;&thinsp; if(Merge) update patch list  

4. ```PatchTree``` reduce of sub fields
5. rebuild local table
```

Implementation : 

```c++

void scheduler_step(){

    // update patch list  
    patch_list.sync_global();

    // rebuild patch index map
    patch_list.build_global_idx_map();

    // apply reduction on leafs and corresponding parents
    patch_tree.partial_values_reduction(
        patch_list.global, 
        patch_list.id_patch_to_global_idx);

    // Generate merge and split request  
    

    // apply split requests


    // update PatchTree

    // update packing index
    // update patch list

    // generate LB change list 
    std::vector<std::tuple<u64, i32, i32,i32>> change_list = 
        make_change_list(patch_list.global);

    // apply LB change list
    patch_data.apply_change_list(change_list, patch_list);


    // apply merge requests  
    // update PatchTree
    // if(Merge) update patch list  

    //rebuild local table
    owned_patch_id = patch_list.build_local();
}

```