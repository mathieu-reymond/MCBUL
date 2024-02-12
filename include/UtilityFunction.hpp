#ifndef UTILITY_FUNCTION_HEADER_FILE
#define UTILITY_FUNCTION_HEADER_FILE

#include <Types.hpp>

class UtilityFunctionPosterior;
class UtilityFunctionParticlePosterior;

class UtilityFunction {
    public:
        UtilityFunction(Vector weights, double noise);
        UtilityFunction(const UtilityFunctionPosterior & posterior, double noise, InitType);
        UtilityFunction(const UtilityFunctionParticlePosterior & posterior, double noise, InitType);

        void reset(const UtilityFunctionPosterior & posterior, InitType);
        void reset(const UtilityFunctionParticlePosterior & posterior, InitType);

        double scalarizeReward(const Vector & rewards) const;
        bool evaluateDiff(const Vector & diff) const;

        const Vector & getWeights() const;

    private:
        Vector weights_;
        double noise_;
        mutable std::mt19937 rand_;
};

#endif
