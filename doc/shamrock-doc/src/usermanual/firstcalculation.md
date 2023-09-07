# Running your first calculation
First you need to set up a configuration:
```bash
./shamrock --sycl-cfg 0:0 --sycl-ls-map
mpirun -n 4 ./shamrock --sycl-cfg 0:0 --sycl-ls-map
```
## Using an ipython console within the terminal

First make sure you have ipython installed:
```bash
sudo apt install python3-ipython
```
Then open the console

```bash
./shamrock --sycl-cfg 0:0 --sycl-ls-map --ipython
```
### run a pre-written script



```bash
./shamrock --sycl-cfg 0:0 --sycl-ls-map --loglevel 10 --rscript ../exemples/spherical_wave.py 
```








