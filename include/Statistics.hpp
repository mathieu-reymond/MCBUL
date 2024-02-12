#ifndef STATISTICS_HEADER_FILE
#define STATISTICS_HEADER_FILE

#include <random>
#include <Types.hpp>

std::pair<Float, Float> sampleNormalPosterior(Float sampleMu, Float sampleSumOfSquares, Float n, std::mt19937 & rnd, Float priorMu = 0.0, Float priorN = 1e-8, Float alpha = 0.5, Float beta = 0.1);

Float sampleMeanPosterior(Float sampleMu, Float sampleSumOfSquares, unsigned n, std::mt19937 & rnd);

void sampleMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd);
void sampleMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd);

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & cov, Vector & out, std::mt19937 & rnd);
void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd);

/**
 * @brief This function samples an index from a probability vector.
 *
 * This function randomly samples an index between 0 and d, given a
 * vector containing the probabilities of sampling each of the indexes.
 *
 * For performance reasons this function does not verify that the input
 * container is effectively a probability.
 *
 * The generator has to be supplied to the function, so that different
 * objects are able to maintain different generators, to reduce correlations
 * between different samples. The generator has to be compatible with
 * std::uniform_real_distribution<Float>, since that is what is used
 * to obtain the random sample.
 *
 * @tparam T The type of the container vector to sample.
 * @tparam G The type of the generator used.
 * @param in The external probability container.
 * @param d The size of the supplied container.
 * @param generator The generator used to sample.
 *
 * @return An index in range [0,d-1].
 */
size_t sampleProbability(const Vector& in, std::mt19937& generator);

/**
 * @brief This function generates a random probability vector.
 *
 * This function will sample uniformly from the simplex space with the
 * specified number of dimensions.
 *
 * S must be at least one or we don't guarantee any behaviour.
 *
 * @param S The number of entries of the output vector.
 * @param generator A random number generator.
 *
 * @return A new random probability vector.
 */
Vector makeRandomProbability(const size_t S, std::mt19937 & generator);

/**
 * @brief This class represents the Alias sampling method.
 *
 * This is an O(1) way to sample from a fixed distribution. Construction
 * takes O(N).
 *
 * The class stores two vectors of size N, and converts the input
 * probability distribution into a set of N weighted coins, each of which
 * represents a choice between two particular numbers.
 *
 * When sampled, the class simply decides which coin to use, and it rolls
 * it. This is much faster than the sampleProbability method, which is
 * O(N), as it needs to iterate over the input probability vector.
 *
 * This is the preferred method of sampling for distributions that
 * generally do not change (as if the distribution changes, the instance of
 * VoseAliasSampler must be rebuilt).
 */
class VoseAliasSampler {
    public:
        /**
         * @brief Basic constructor.
         *
         * @param p The probability distribution to sample from (normalized).
         */
        VoseAliasSampler(const Vector & p);

        /**
         * @brief This function samples a number that follows the distribution of the class.
         *
         * @param generator A random number generator.
         *
         * @return A number between 0 and the size of the original ProbabilityVector.
         */
        template <typename G>
        size_t sampleProbability(G & generator) const {
            const auto x = sampleDistribution_(generator);
            const int i = x;
            const auto y = x - i;

            if (y < prob_[i]) return i;
            return alias_[i];
        }

    private:
        Vector prob_;
        std::vector<size_t> alias_;
        mutable std::uniform_real_distribution<Float> sampleDistribution_;
};

#endif
