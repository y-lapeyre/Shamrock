# New TODOs : 

    - Tree Field for multipoles
    - implement tree send
    - implement tree field send

    - Implement full tree field (aka patch field + radix tree field)


# Endgame : 

Scheduler : Move data

Full tree = Patch Octree (Radix tree)
full tree field = Patch field X Radix tree field

interaction condition = interact cond for full tree (eventually using n full tree field)

based on the interaction condition : 
    full walk = walk in patch octree and then on radix tree if unroll
    interface handler prepare data for a full walk





# Patch Scheduler : 

- [ ] patchdata_isend -> patchdata::pisend 
- [ ] patchdata_irecv -> patchdata::pirecv
- [ ] Builder for SchedulerMPI since it won't work with no patches
- [ ] put instance of simbox in the SchedulerMPI
- [ ] runtime error in SchedulerMPI construction if patch type inactive
- [ ] make SimulationBox templated and store tranlate & scale factor
- [ ] for the templated patch do something like shown in main.cpp
- [ ] SerialPatchTree attach buf by default and use accesors
- [ ] Transform box info into boundary condition stuff

# Hardware stuff : 
- [ ] Create hardware class to detect environment & handle MPI + sycl
- [ ] move init/free required type in SchedulerMPI to patch & sycl interop class 
- [ ] move sycl interop init in mpi_handler
- [ ] namedtimer destructor = end if not already stopped


# Leapfrog : 
- [ ] clean the leapfrog

# Tree : 
- [ ] check validity of one cell mode

# Sycl :
- [ ] move sycl kernels to namespaces instead of ::

# Global : 
- [ ] use only patch_id or id_patch not both
- [ ] track uses of max_box_sz in the code (should find a way to abstract it)
- [ ] instance __compute_object_patch_owner with f32/f64 modes (compile time reduction)
- [ ] implement the interaction box computation of the radix tree using U1 buf instead of a separate buf
- [ ] in interface split the get_flag_id part and cache it to avoid recomputing it


