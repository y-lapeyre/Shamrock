# Shamrock 101: basics of the code

Everything you'll want to touch is in the src folder. Quite intuitively Shammodels hosts all of the models implemented in Shamrock. 
Lets consider the SPH model. All that is contained in the cpp file will be compiled with Shamrock. Hence only essential parts of the alogrithm (functions called many many times) go there, and you probably don't need to go in there.

Some functions are backend depended: for instance, how the min between two integers is computed using different algorithm on cuda or openCL. In order to not be bothered by that while coding, Shamrock uses SYCL. All the functions that are backend dependent go in .... namespace. This namespace also contains an alias to base functions that are independent of the backend. This is because vectorizing bbase functions depends on the backend?

