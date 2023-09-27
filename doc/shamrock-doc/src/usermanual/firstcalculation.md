# Running your first calculation
First you need to set up a SYCL configuration:

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
## run a pre-written script
Setting and running a simulation can all be done through a python script. You don't need to go through the arcanes of SHAMROCK! 
SHAMROCK provides an array of pre-cooked scripts you can run as is. When running a simulation of your own, you can just follow the same steps as for the pre-cooked scripts.
For the sake of clarity, let's take the example of the spherical_wave.py script. To run it, type this in your terminal:

```bash
./shamrock --sycl-cfg 0:0 --sycl-ls-map --loglevel 10 --rscript ../exemples/spherical_wave.py 
```
the --loglevel argument specifies the degree of verbosity of SHAMROCK.
0 silent
1 tells basic steps
10 good for basic use
100 specifies instructions in the kernel

## importing a setup from phantom








