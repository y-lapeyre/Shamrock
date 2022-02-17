# MPI Scheduler in SHAMROCK





```{prf:algorithm} MPIScheduler
1. Initialisation  
    &ensp;&thinsp; update patch list  
    &ensp;&thinsp; Generate merge and split request  

2. Patch Splitting  
    &ensp;&thinsp; apply split requests  
    &ensp;&thinsp; update patchtree

2. Patch Merging & LB  
    &ensp;&thinsp; update packing index  
    &ensp;&thinsp; Load balancing  
    &ensp;&thinsp; apply merge requests  
    &ensp;&thinsp; update patchtree  
    &ensp;&thinsp; if(Merge) update patch list  
```