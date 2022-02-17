# SchedulerMPI





```{prf:algorithm} MPIScheduler
1. Initialisation  
    &ensp;&thinsp; update patch list  
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
```