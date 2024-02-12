#include <UtilityFunction.hpp>

#include <Seeder.hpp>
#include <UtilityFunctionPosterior.hpp>
#include <UtilityFunctionParticlePosterior.hpp>
#include <Statistics.hpp>

UtilityFunction::UtilityFunction(Vector weights, double noise) :
        weights_(std::move(weights)), noise_(noise), rand_(Seeder::getSeed()) {}

UtilityFunction::UtilityFunction(const UtilityFunctionPosterior & upost, double noise, InitType i) :
        weights_(upost.getW()), noise_(noise), rand_(Seeder::getSeed())
{
    reset(upost, i);
}

UtilityFunction::UtilityFunction(const UtilityFunctionParticlePosterior & upost, double noise, InitType i) :
        weights_(upost.getW()), noise_(noise), rand_(Seeder::getSeed())
{
    reset(upost, i);
}

void UtilityFunction::reset(const UtilityFunctionPosterior & upost, InitType i) {
    if (i == InitType::ML) {
        weights_ = upost.getMean();
    } else {
        sampleNormalizedMultivariateNormalInline(upost.getMean(), upost.getCovLLT(), weights_, rand_);
    }
}

void UtilityFunction::reset(const UtilityFunctionParticlePosterior & upost, InitType i) {
    if (i == InitType::ML) {
        weights_ = upost.getMean();
    } else {
        size_t weightsID = sampleProbability(upost.getWeights(), rand_);
        weights_ = upost.getParticles().row(weightsID);
    }
}

double UtilityFunction::scalarizeReward(const Vector & rewards) const {
    return rewards.dot(weights_);
}

bool UtilityFunction::evaluateDiff(const Vector & diff) const {
    return diff.dot(weights_) >= 0;
}

const Vector & UtilityFunction::getWeights() const {
    return weights_;
}
