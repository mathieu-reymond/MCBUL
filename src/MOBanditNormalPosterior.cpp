#include <MOBanditNormalPosterior.hpp>

#include <MOBandit.hpp>
#include <Statistics.hpp>

MOBanditNormalPosterior::MOBanditNormalPosterior(const size_t A, const size_t W, bool saveSamples) :
        sampleMu_(A, W), sampleSumOfSquares_(A, W), counts_(A),
        saveSamples_(saveSamples)
{
    sampleMu_.setZero();
    sampleSumOfSquares_.setZero();

    if (saveSamples_) {
        samples_.reserve(A);
        for (size_t a = 0; a < A; ++a)
            samples_.emplace_back(15, W);
    }
}

MOBanditNormalPosterior::MOBanditNormalPosterior(const MOBandit & bandit, bool saveSamples) :
        MOBanditNormalPosterior(bandit.getA(), bandit.getW(), saveSamples) {}

void MOBanditNormalPosterior::record(const size_t a, const MOBandit & bandit) {
    record(a, bandit.sampleR(a));
}

void MOBanditNormalPosterior::record(const size_t a, const Vector & sample) {
    ++counts_[a];

    for (size_t o = 0; o < getW(); ++o) {
        const auto delta = sample[o] - sampleMu_(a, o);
        sampleMu_(a, o) += delta / counts_[a];
        sampleSumOfSquares_(a, o) += delta * (sample[o] - sampleMu_(a, o));
    }

    if (saveSamples_)
        samples_[a].push_back(sample);
}

std::pair<Float, Float> MOBanditNormalPosterior::sampleMeanStd(size_t a, size_t o, std::mt19937 & rnd) const {
    return sampleNormalPosterior(sampleMu_(a, o), sampleSumOfSquares_(a, o), counts_[a], rnd);
}

Float MOBanditNormalPosterior::sampleMean(size_t a, size_t o, std::mt19937 & rnd) const {
    return sampleMeanPosterior(sampleMu_(a, o), sampleSumOfSquares_(a, o), counts_[a], rnd);
}

std::pair<Float, Float> MOBanditNormalPosterior::maxLikelihoodMeanStd(size_t a, size_t o) const {
    // Compute ~unbiased estimator for normal std (NOT VARIANCE)
    // We divide by (N-1.5) rather than (N-1)
    Float unbiasedStd = sampleSumOfSquares_(a, o) / (counts_[a] - 1.5);
    // Avoid possible negative values
    unbiasedStd = std::sqrt(std::max((Float) 0, unbiasedStd));

    return {sampleMu_(a, o), unbiasedStd};
}

size_t MOBanditNormalPosterior::getA() const { return sampleMu_.rows(); }
size_t MOBanditNormalPosterior::getW() const { return sampleMu_.cols(); }
const Matrix2D & MOBanditNormalPosterior::getMeans() const { return sampleMu_; }
const Matrix2D & MOBanditNormalPosterior::getSampleSquares() const { return sampleSumOfSquares_; }
const std::vector<unsigned> & MOBanditNormalPosterior::getCounts() const { return counts_; }
const std::vector<StorageMatrix2D> & MOBanditNormalPosterior::getSamples() const { return samples_; }
