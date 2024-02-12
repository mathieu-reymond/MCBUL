#ifndef PAIR_INTERLEAVED_ROLLOUT_HEADER_FILE
#define PAIR_INTERLEAVED_ROLLOUT_HEADER_FILE

#include <random>
#include <Types.hpp>
#include <Seeder.hpp>
#include <iostream>

template <typename Env>
class PairInterleaved {
    public:
        PairInterleaved();
        double continueRollout(Env& env);
    private:
        std::vector<size_t> arms_;
        std::mt19937 engine_;
};

template <typename Env>
PairInterleaved<Env>::PairInterleaved() : engine_(Seeder::getSeed()) {}

template <typename Env>
double PairInterleaved<Env>::continueRollout(Env& env) {
    int obs;
    double reward = 0;
    bool terminal;

    // get budgets, to evenly spread out arm pull
    auto [b1, b2] = env.getBudgets();
    auto [s1, s2] = env.getSteps();
    double queryWait = static_cast<double>(b1) / b2;
    double currentQueryWait = s1 - std::floor(s2*queryWait);

    b1 -= s1;
    b2 -= s2;

    while (b1 > 0 || b2 > 0) {
        currentQueryWait += 1.0;
        size_t action = 0;
        if (b2 && currentQueryWait >= queryWait)
            action = 1;

        std::tie(obs, reward, terminal) = env.step(action);
        if (terminal)
            return reward;
        if (action) {
            --b2;
            currentQueryWait -= queryWait;
        }
        else --b1;
    }

    return reward;
}

#endif

