#include <Statistics.hpp>

#include <cassert>
#include <random>

static std::uniform_real_distribution<Float> pDist(0.0, 1.0);

std::pair<Float, Float> sampleNormalPosterior(Float sampleMu, Float sampleSumOfSquares, Float n, std::mt19937 & rnd, Float priorMu, Float priorN, Float alpha, Float beta) {
    std::gamma_distribution<Float> gd(alpha + n/2.0,
                                      1.0/(
                                           beta +
                                           sampleSumOfSquares/2.0 +
                                           (n * priorN * std::pow(sampleMu-priorMu, 2.0))/(2.0 * (n+priorN))
                                        )
                                      );
    Float tau = gd(rnd);
    std::normal_distribution<Float> nd(n/(n+priorN)*sampleMu + priorN/(n+priorN)*priorMu,
                                       std::sqrt(1/(tau*(priorN+n))));

    Float mu = nd(rnd);
    Float sigma = std::sqrt(1/tau);
    return {mu, sigma};
}

Float sampleMeanPosterior(Float sampleMu, Float sampleSumOfSquares, unsigned n, std::mt19937 & rnd) {
    static std::student_t_distribution<Float> dist;

    assert(n > 1);
    if (dist.n() != n)
        dist = std::student_t_distribution<Float>(n - 1);

    //     mu = est_mu - t * s / sqrt(n)
    // where
    //     s^2 = 1 / (n-1) * sum_i (x_i - est_mu)^2
    // and
    //     t = student_t sample with n-1 degrees of freedom
    return sampleMu + dist(rnd) * std::sqrt(sampleSumOfSquares / (n * (n - 1)));
}

void sampleMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd) {
    static std::normal_distribution<Float> dist;
    auto randN = [&rnd](){ return dist(rnd); };

    out = mu + llt.matrixL() * Vector::NullaryExpr(mu.size(), randN);
}

void sampleMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd) {
    // LLT is basically to square root the variance (since we need the std).
    sampleMultivariateNormalInline(mu, cov.llt(), out, rnd);
}

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Eigen::LLT<Matrix2D> & llt, Vector & out, std::mt19937 & rnd) {
    sampleMultivariateNormalInline(mu, llt, out, rnd);

    // Normalize (note the abs!)
    out.array() /= out.array().abs().sum();
}

void sampleNormalizedMultivariateNormalInline(const Vector & mu, const Matrix2D & cov, Vector & out, std::mt19937 & rnd) {
    sampleNormalizedMultivariateNormalInline(mu, cov.llt(), out, rnd);
}

size_t sampleProbability(const Vector& in, std::mt19937& generator) {
    const size_t D = in.size();
    Float p = pDist(generator);

    for ( size_t i = 0; i < D; ++i ) {
        if ( in[i] > p ) return i;
        p -= in[i];
    }
    return D-1;
}

Vector makeRandomProbability(const size_t S, std::mt19937 & generator) {
    Vector b(S);
    Float * bData = b.data();
    // The way this works is that we're going to generate S-1 numbers in
    // [0,1], and sort them with together with an implied 0.0 and 1.0, for
    // a total of S+1 numbers.
    //
    // The output will be represented by the differences between each pair
    // of numbers, after sorting the original vector.
    //
    // The idea is basically to take a unit vector and cut it up into
    // random parts. The size of each part is the value of an entry of the
    // output.

    // We must set the first element to zero even if we're later
    // overwriting it. This is to avoid bugs in case the input S is one -
    // in which case we should return a vector with a single element
    // containing 1.0.
    bData[0] = 0.0;
    for ( size_t s = 0; s < S-1; ++s )
        bData[s] = pDist(generator);

    // Sort all but the implied last 1.0 which we'll add later.
    std::sort(bData, bData + S - 1);

    // For each number, keep track of what was in the vector there, and
    // transform it into the difference with its predecessor.
    Float helper1 = bData[0], helper2;
    for ( size_t s = 1; s < S - 1; ++s ) {
        helper2 = bData[s];
        bData[s] -= helper1;
        helper1 = helper2;
    }
    // The final one is computed with respect to the overall sum of 1.0.
    bData[S-1] = 1.0 - helper1;

    return b;
}

VoseAliasSampler::VoseAliasSampler(const Vector & p) :
        prob_(p), alias_(prob_.size()), sampleDistribution_(0, prob_.size())
{
    // Here we do the Vose Alias setup in a way that avoids the creation of
    // the small and large arrays.
    //
    // In practice what we do is we keep two pointers, one for large
    // elements and one for the small ones, and we move them along the
    // array as if we had already sorted the thing.

    const auto avg = 1.0 / prob_.size();
    auto small = 0, large = 0;
    while (small < prob_.size() && prob_[small] >= avg) ++small;
    while (large < prob_.size() && prob_[large] < avg) ++large;

    auto smallCheckpoint = small;

    while (small < prob_.size() && large < prob_.size()) {
        // Note: we do not do any assignments to prob_[small] here since if
        // we scaled the values already we might trip the large counter (as
        // it might be behind the small counter).
        prob_[large] = (prob_[large] + prob_[small]) - avg;
        alias_[small] = large;

        // If the large became small, we temporarily move the small counter
        // here, and look around for a new large element.
        // Otherwise, we go back to our last small 'checkpoint', and we
        // look for a new small element.
        if (prob_[large] < avg) {
            small = large;
            ++large;
            while (large < prob_.size() && prob_[large] < avg) ++large;
        } else {
            small = smallCheckpoint + 1;
            while (small < prob_.size() && prob_[small] >= avg) ++small;
            // Set the checkpoint again
            smallCheckpoint = small;
        }
    }

    // Now, for each entry which remained unassigned (so it is still with
    // the 0 default in the alias vector), we set it to just reference
    // itself. This takes care of both large and small entries which have
    // been left with no pairings.
    auto x = std::min(large, small);
    while (x < prob_.size()) {
        prob_[x] = 1.0;
        alias_[x] = x;
        ++x;
        while (x < prob_.size() && alias_[x] != 0) ++x;
    }

    // Here we scale up the vector so that each entry can be correctly seen
    // as a weighted coin. Note that all 1.0 entries will now be larger,
    // but for those there's no choice so we don't care about the precise
    // value anyway.
    prob_ *= prob_.size();
}
