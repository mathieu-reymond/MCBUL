How To
======

Requirements
------------

The project requires:

- CMake
- Compiler support for C++17 (so at least `g++-7`)
- Boost library
- The Eigen matrix library 3.3.
- Python3

Instructions
------------

Compile the project (please name the build folder `build` as the experiment
running/plotting code expects it like that).

```
mkdir build
cd build
cmake ..
make
cd ..
```

Running the baselines
---------------------

The baselines (multi-objective top-two Thompson sampling) are available in a separate repository:
<https://github.com/mathieu-reymond/MOTTTS>

Please follow the instructions from the repo to run the baselines.

Running the code
----------------

In the [baseline repository](https://github.com/mathieu-reymond/MOTTTS), we have pre-generated 1Ok+ multi-objective multi-armed bandits (MOMABs). These are situated in the `generated_bandits` directory.

To run MCBUL on any of these generated bandits, run, e.g.,:

```
./build/src/main --rollouts 10000 --bandit-budget 20 --oracle-budget 3 --score-samples 1 --multiplier 1 --experiments 5 --file <PATH_TO_MOTTTS>/generated_bandits/run_3_bandit_1.txt
```
