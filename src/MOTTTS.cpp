#include <MOTTTS.hpp>

size_t bestAction(const MOBanditNormalPosterior& bpost, const Vector& uf, std::mt19937 & rnd) {
    size_t bestA = 0;
    double bestU = -std::numeric_limits<float>::infinity();
    for (size_t a = 0; a < bpost.getA(); ++a) {
        double utility = 0;
        for (size_t o = 0; o < bpost.getW(); ++o) {
            utility += bpost.sampleMean(a, o, rnd) * uf[o];
        }
        if (utility > bestU) {
            bestA = a;
            bestU = utility;
        }
    }
    return bestA;
}

size_t recommendPull(const MOBanditNormalPosterior& bpost, const Vector& uf, double beta, std::mt19937 & rnd) {
    size_t bestA = bestAction(bpost, uf, rnd);
    // flip a coin to decide if you recommend best or second-best action
    std::bernoulli_distribution pickBest(beta);
    if (pickBest(rnd))
        return bestA;
    size_t secondA;
    do {
        secondA = bestAction(bpost, uf, rnd);
    } while (secondA == bestA);
    return secondA;
}
