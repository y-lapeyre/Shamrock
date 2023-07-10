# Neowise

```bash
rm -rf Shamrock
export DPCPP_HOME=$(pwd)/dpcpp_compiler
export PATH=$DPCPP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH
tar -xvf Shamrock.tar.gz 
cd Shamrock
python3 buildbot/configure.py --gen make --tests --build release --outdir dpcpp_rocm --cxxpath ../llvm/build --compiler dpcpp --profile hip-gfx906 --cxxflags="--rocm-path=/opt/rocm"
cd dpcpp_rocm
make -j
```