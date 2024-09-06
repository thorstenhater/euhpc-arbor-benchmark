# Arbor EUHPC Mini-Benchmark

Use Arbor in the CPM branch here:

https://github.com/thorstenhater/arbor/tree/cmake/cpm

Ensure CMake, Python, Ninja, MPI, hwloc, and CUDA are in scope.

Build Arbor
``` sh
git clone --branch cmake/cpm https://github.com/thorstenhater/arbor.git
mkdir arbor/build
cd arbor/build
# Note: adjust CUDA Arch as needed (80 = A100, 90 = H100)
# Note: adjust installation path to taste
CXX=mpicxx CC=mpicc cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DARB_WITH_MPI=ON -DARB_GPU=cuda -DARB_WITH_PYTHON=OFF -GNinja -DARB_VECTORIZE=ON -DCMAKE_INSTALL_PREFIX=$HOME -DARB_USE_HWLOC=ON
cd ~
```

Now the benchmark
``` sh
git clone https://github.com/thorstenhater/euhpc-arbor-benchmark.git bench
mkdir bench/build
cd bench/build
CXX=mpicxx CC=mpicc cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -G Ninja
```

Generate input and SLURM files
``` sh
# Note: adjust values in script first for cell and node counts, accounting, etc.
python ../gen-inputs.py
```


Run benchmark, for example
``` sh
sbatch submit-cells=128000-nodes=4.job
```


