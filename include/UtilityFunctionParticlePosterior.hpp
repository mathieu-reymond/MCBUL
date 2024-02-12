#ifndef UTILITY_FUNCTION_PARTICLE_POSTERIOR_HEADER_FILE
#define UTILITY_FUNCTION_PARTICLE_POSTERIOR_HEADER_FILE

#include <StorageEigen.hpp>

class UtilityFunction;
class MOBanditNormalPosterior;

class UtilityFunctionParticlePosterior {
    public:
        UtilityFunctionParticlePosterior(const Matrix2D & particles, double noise);
        UtilityFunctionParticlePosterior(const UtilityFunctionParticlePosterior &);
        UtilityFunctionParticlePosterior & operator=(const UtilityFunctionParticlePosterior & other);

        void record(const Vector & diff, bool answer);
        void record(const Vector & diff, const UtilityFunction & uf);

        const Matrix2D & getParticles() const;
        const Vector & getWeights() const;
        Vector getMean() const;
        size_t getW() const;
        double getNoise() const;

    private:
        const Matrix2D & particles_;
        Vector weights_;
        double noise_;
};

const Vector & suggestQueryThompson(const MOBanditNormalPosterior & bpost, const UtilityFunctionParticlePosterior & upost);

#endif

