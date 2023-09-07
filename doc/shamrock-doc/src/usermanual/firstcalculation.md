
./shamrock --sycl-cfg 0:0 --sycl-ls-map
mpirun -n 4 ./shamrock --sycl-cfg 0:0 --sycl-ls-map

./shamrock --sycl-cfg 0:0 --sycl-ls-map --ipython
sudo apt install python3-ipython

./shamrock --sycl-cfg 0:0 --sycl-ls-map --loglevel 10 --rscript ../exemples/spherical_wave.py 

