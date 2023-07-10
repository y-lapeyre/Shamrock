# Running on cluster

## Neowise
10 nodes, 10 cpus, 480 cores, 8 gpu per nodes

```bash
export DPCPP_HOME=$(pwd)/dpcpp_compiler
export PATH=$DPCPP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH

tar -xvf Shamrock.tar.gz 
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --outdir dpcpp_rocm --cxxpath ../llvm/build --compiler dpcpp --profile hip-gfx906 --cxxflags="--rocm-path=/opt/rocm"
cd dpcpp_rocm
make -j
```

Runing the code on 8 gpu per nodes
```bash
mpirun -machinefile $OAR_NODEFILE -npernode 8 -x PATH=~/dpcpp_compiler/bin:$PATH -x LD_LIBRARY_PATH=~/dpcpp_compiler/lib:$LD_LIBRARY_PATH ./shamrock --sycl-cfg auto:HIP --loglevel 1 --sycl-ls-map  --rscript ../exemples/spherical_wave.py
```