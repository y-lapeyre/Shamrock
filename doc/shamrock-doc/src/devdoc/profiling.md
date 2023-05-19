# Profiling

In the code many function starts with 

```c++
StackEntry stack_loc{};
```
the profiling can be disabled for a function by using :
```c++
StackEntry stack_loc{false};
```

This is used to trace the location in the code, allowing more precise error message, but also profiling !

when the code is ran some files name `timings_*` will be created, they hold the flame graph tracing of the code.

```bash
python buildbot/merge_profilings.py timings_* merge_prof                          
```

This will merged the output of each files.

Go on `https://ui.perfetto.dev/` and upload the `merge_prof` file you will see the actual trace of the code execution.

