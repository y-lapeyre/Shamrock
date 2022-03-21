# Compiling

## Supported backends

DPCPP : 
| Backend | Supported | arch |
| --- | ---  | ---  |
| CUDA | V  | x86 |
| OpenCL | X  |  |

HipSYCL : 
| Backend | Supported | arch |
| --- | ---  | ---  |
| OpenMP | V  | x86 & amr64 |

## Setup SYCL

```bash
wget https://gitlab.com/tdavidcl/sycl-install-script/-/raw/main/setup_sycl.sh
sh setup_sycl.sh
```

if you plan on using dpcpp run in terminal : 
```bash
export DPCPP_HOME=<path>/sycl_cpl/dpcpp
export PATH=$DPCPP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/lib:$LD_LIBRARY_PATH
```


## Compilation



pull the directory : 
```bash
git pull https://gitlab.com/tdavidcl/shamrock.git
```

```bash
cd shamrock/buildbot
```

configure utility : 

```bash
python3 configure.py --interactive ../../sycl_cpl/hipsycl
```
```bash
python3 configure.py --interactive ../../sycl_cpl/dpcpp
```




```bash
python3 compile.py
```