#include <Bucketing.hpp>

#include <MOBanditNormalPosterior.hpp>

size_t RewardBucketer::getW() const {
    return (precision+1) * (precision+1);
}

size_t RewardBucketer::operator()(const Vector & rew, const MOBanditNormalPosterior & bpost, size_t a) {
    const Float N = bpost.getCounts()[a];

    size_t discretized = 0;
    for (size_t o = 0; o < bpost.getW(); ++o) {
        // bin from 0 to precision*4 included
        const Float sMu = bpost.getMeans()(a, o);
        // alpha=0.5
        const Float k = 0.5 + N/2.0;
        // beta=0.1, priorN=1e-8, priorMu=0
        const Float tau = 1.0/(0.1 + bpost.getSampleSquares()(a, o)/2.0 + (N*1e-8*std::pow(sMu, 2.0))/(2.0*(N+1e-8)));
        // Mean of gamma: k * tau == mean precision
        const Float std = std::sqrt(1.0/(k*tau))*1.5;
        const Float lowerBound = sMu - std;

        const size_t bin = std::lround(precision*(std::clamp(rew[o], lowerBound, sMu + std) - lowerBound)/(std*2.0));

        discretized += std::pow(precision+1, o) * bin;
    }
    return discretized;
}
