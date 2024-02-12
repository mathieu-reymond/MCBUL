#include <UtilityFunctionParticlePosterior.hpp>

#include <Seeder.hpp>
#include <Statistics.hpp>
#include <MOBandit.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <UtilityFunction.hpp>

UtilityFunctionParticlePosterior::UtilityFunctionParticlePosterior(const Matrix2D & particles, double noise) :
        particles_(particles), weights_(Vector::Constant(particles_.rows(), 1.0/particles_.rows())), noise_(noise) {}


UtilityFunctionParticlePosterior::UtilityFunctionParticlePosterior(const UtilityFunctionParticlePosterior & other) :
        particles_(other.particles_), weights_(other.weights_), noise_(other.noise_)
{
    assert(getW() == other.getW());
}

UtilityFunctionParticlePosterior & UtilityFunctionParticlePosterior::operator=(const UtilityFunctionParticlePosterior & other) {
    assert(getW() == other.getW());

    // assume particles are the same, they won't change
    // particles_ = other.particles_;
    weights_ = other.weights_;
    noise_ = other.noise_;

    return *this;
}

void UtilityFunctionParticlePosterior::record(const Vector & diff, bool answer) {
    // Equivalent to w *= v*(1-n) + (1-v)*n
    weights_.array() *= (1.0 - 2.0 * noise_) * (((particles_ * diff).array() >= 0.0) == answer).cast<Float>() + noise_;
    // Normalize
    weights_.array() /= weights_.sum();
}

void UtilityFunctionParticlePosterior::record(const Vector & diff, const UtilityFunction & uf) {
    record(diff, uf.evaluateDiff(diff));
}

Vector UtilityFunctionParticlePosterior::getMean() const {
    return (particles_.array().colwise() * weights_.array()).colwise().sum();
}

size_t UtilityFunctionParticlePosterior::getW() const { return particles_.cols(); }
const Matrix2D & UtilityFunctionParticlePosterior::getParticles() const { return particles_; }
const Vector & UtilityFunctionParticlePosterior::getWeights() const { return weights_; }
double UtilityFunctionParticlePosterior::getNoise() const { return noise_; }

/// ###################################################

const Vector & suggestQueryThompson(const MOBanditNormalPosterior & bpost, const UtilityFunctionParticlePosterior & upost) {
    static Vector retval, bestArm, bestArm2;
    static std::mt19937 rnd(Seeder::getSeed());

    retval.resize(bpost.getW());
    bestArm.resize(bpost.getW());
    bestArm2.resize(bpost.getW());

    double bestArmV, bestArm2V;
    bestArmV = bestArm2V = -std::numeric_limits<double>::infinity();

    const auto & particles = upost.getParticles();

    for (size_t i = 0; i < 100; ++i) {
        // Sample UF
        size_t pId = sampleProbability(upost.getWeights(), rnd);

        // Sample each arm, and remember best two.
        for (size_t a = 0; a < bpost.getA(); ++a) {
            for (size_t o = 0; o < bpost.getW(); ++o)
                retval[o] = bpost.sampleMean(a, o, rnd);

            double utility = retval.dot(particles.row(pId));
            if (utility > bestArm2V) {
                if (utility > bestArmV) {
                    bestArm2V = bestArmV;
                    bestArm2 = bestArm;

                    bestArmV = utility;
                    bestArm = retval;
                } else {
                    bestArm2V = utility;
                    bestArm2 = retval;
                }
            }
        }

        // Try again until we have an informative query to ask.
        if (!dominates(bestArm, bestArm2))
            break;
    }

    retval = bestArm - bestArm2;
    return retval;
}
