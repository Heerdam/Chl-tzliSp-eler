
#define USE_AVX
#define GPU_NEIGHBORHOOD_SEARCH

#include "SPlisHSPlasH/Common.h"
#include <SPlisHSPlasH/Simulation.h>

#include <Utilities/Logger.h>
#include <Utilities/Timing.h>
#include <Utilities/Counting.h>

INIT_LOGGING
INIT_TIMING
INIT_COUNTING

int main() {

    SPH::Simulation sim;
    SPH::Simulation::setCurrent(&sim);

    sim.init(0.1, false);

    sim.deferredInit();



    




    return EXIT_SUCCESS;
}