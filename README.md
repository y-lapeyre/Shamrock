
<img src="doc/logosham_white.png" alt="logo" width="600"/>


# Getting started

```bash
git clone git@github.com:tdavidcl/Shamrock.git
```

## SYCL configuration

```bash
/bin/bash -c "$(curl -fsSL  https://raw.githubusercontent.com/tdavidcl/sycl-setup-script/main/setup_sycl.sh)"
```

If you plan on using dpcpp : 

```bash
export DPCPP_HOME=$(pwd)/sycl_cpl/dpcpp/
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
```

## Compiling the code

```bash
python buildbot/configure.py --interactive (path to compiler)
python buildbot/compile.py
```

To run : 
```bash
export DPCPP_HOME=~/Documents/these/codes/sycl_workspace
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
```


## Directory structure


>- [<span style="color:orange">src</span>]
>    - <span style="color:lightblue">CMakeLists.txt</span> (define flags such as HAS_AMR >
HAS_SPH HAS_TEST HAS_PERIODIC)
>    - <span style="color:lightblue">main_sph.cpp</span> (#ifdef HAS_SPH)
>    - <span style="color:lightblue">main_amr.cpp</span> (#ifdef HAS_AMR)
>    - <span style="color:lightblue">main_visu.cpp</span> (#ifdef HAS_VISU)
>    - <span style="color:lightblue">main_setup_sph.cpp</span> (#ifdef HAS_SPH)
>    - <span style="color:lightblue">main_setup_amr.cpp</span> (#ifdef HAS_AMR)
>    - [<span style="color:orange">tree</span>]
>        - all stuff for the general tree
>    - [<span style="color:orange">phys</span>]
>        - eos
>        - eq
>    - [<span style="color:orange">sph</span>]
>    - [<span style="color:orange">amr</span>]
>    - [<span style="color:orange">sys</span>]
>        - <span style="color:lightblue">mpi_handler.cpp</span>
>    - [<span style="color:orange">io</span>]
>        - unifed hdf5 output for both sph and amr (or vtk for amr to discus)
>        - snapshot
>        - setup files
>        - input files
>        - log
>    - [<span style="color:orange">test</span>]
>        - reuse the same directory / file layout as src but with corresponding tests 
>            (allow for compilation without the test suite)
>    - [<span style="color:orange">visu</span>]
>        - all code corresponding to visualisation



