cmake_minimum_required (VERSION 3.12) # CMP0069 NEW

############################
##   Experiment Library   ##
############################

# If you need other cpp files to compile for the experiment, enable the
# following library. Remember to add "ExperimentLibrary" as an additional
# parameter to the "AddExperimentXXX" functions (they'll link the
# ExperimentLibrary then through the ARGN parameter).

# Simple example with library made of file1.cpp and file2.cpp:
#
#     add_library(MCBULLibrary
#         ./file1.cpp
#         ./file2.cpp
#     )
#     target_link_libraries(MCBULLibrary ${Boost_LIBRARIES})
#
# Then you can link MCBULLibrary to the experiments added below.

add_library(MCBULLibrary
    ./Seeder.cpp
    ./StorageEigen.cpp
    ./Statistics.cpp
    ./Adam.cpp
    ./BayesianLogisticRegression.cpp
    ./Bucketing.cpp
    ./MOBandit.cpp
    ./MOBanditNormalPosterior.cpp
    ./UtilityFunction.cpp
    ./UtilityFunctionPosterior.cpp
    ./UtilityFunctionParticlePosterior.cpp
    ./PairParticleEnvironment.cpp
    ./ParticleEnvironment.cpp
    ./Environment.cpp
    ./MOTTTS.cpp
    ./IO.cpp
)


############################
## Experiment Declaration ##
############################

# This is a simple example on how you add a new experiment/executable to the
# build script:
#
#    add_executable(exename
#        ./file1.cpp
#        ./file2.cpp
#    )
#    target_link_libraries(exename MCBULLibrary ${Boost_LIBRARIES})
#
# Then go in the build folder and just do make, it will automatically
# re-run CMake and build the new thing.

add_executable(interactive
    ./interactive.cpp
)
target_link_libraries(interactive MCBULLibrary ${Python3_LIBRARIES})

add_executable(main
    ./main.cpp
)
target_link_libraries(main MCBULLibrary ${GITHASH_LIBRARIES})
