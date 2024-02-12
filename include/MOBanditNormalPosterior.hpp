#ifndef MOBANDIT_NORMAL_POSTERIOR_HEADER_FILE
#define MOBANDIT_NORMAL_POSTERIOR_HEADER_FILE

#include <Types.hpp>
#include <StorageEigen.hpp>

class MOBandit;

class MOBanditNormalPosterior {
    public:
        MOBanditNormalPosterior(size_t A, size_t W, bool saveSamples = false);
        MOBanditNormalPosterior(const MOBandit & bandit, bool saveSamples = false);

        void record(size_t arm, const Vector & rew);
        void record(size_t arm, const MOBandit & bandit);

        std::pair<Float, Float> sampleMeanStd(size_t arm, size_t obj, std::mt19937 & rnd) const;
        Float sampleMean(size_t arm, size_t obj, std::mt19937 & rnd) const;

        std::pair<Float, Float> maxLikelihoodMeanStd(size_t arm, size_t obj) const;

        size_t getA() const;
        size_t getW() const;

        const Matrix2D & getMeans() const;
        const Matrix2D & getSampleSquares() const;
        const std::vector<unsigned> & getCounts() const;

        const std::vector<StorageMatrix2D> & getSamples() const;

    private:
        Matrix2D sampleMu_;
        Matrix2D sampleSumOfSquares_;
        std::vector<unsigned> counts_;

        bool saveSamples_;
        std::vector<StorageMatrix2D> samples_;
};

#endif
