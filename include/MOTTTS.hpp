#ifndef MOTTTS_HEADER_FILE
#define MOTTTS_HEADER_FILE

#include <iostream>
#include <limits>
#include <random>
#include <Types.hpp>
#include <Seeder.hpp>

#include <MOBanditNormalPosterior.hpp>

template <typename Env>
class MOTTTS {
    public:
        MOTTTS();
        double continueRollout(Env& env);
        static constexpr double beta = 0.5;

    private:
        std::mt19937 engine_;
};

size_t recommendPull(const MOBanditNormalPosterior& bpost, const Vector& uf, double beta, std::mt19937 & rnd);

template <typename Env>
MOTTTS<Env>::MOTTTS() : engine_(Seeder::getSeed()) {}

template <typename Env>
double MOTTTS<Env>::continueRollout(Env& env) {
    int obs;
    double reward;
    bool terminal;
    size_t action;

    const auto & bpost2 = env.getBandit2Posterior();
    const auto & upost2 = env.getUtilityFunction2Posterior();

    Vector upost2Mean = upost2.getMean();

    // get budgets, determine when to ask queries
    auto [b1, b2] = env.getBudgets();
    auto [s1, s2] = env.getSteps();
    b1 -= s1;
    b2 -= s2;
    double queryWait = b2 == 0 ? std::numeric_limits<float>::infinity() : (b1+b2)/b2;
    double currentQueryWait = 0;
    do {
        currentQueryWait += 1.0;
        if (b2 && currentQueryWait >= queryWait) {
            // choose query action
            action = env.getCurrentA()-1;
            currentQueryWait -= queryWait;
            --b2;
            std::tie(obs, reward, terminal) = env.step(action);
            // get updated ufposterior mean
            upost2Mean = upost2.getMean();
        } else {
            action = recommendPull(bpost2, upost2Mean, beta, engine_);
            std::tie(obs, reward, terminal) = env.step(action);
        }

    } while (!terminal);

    return reward;
}

#endif
