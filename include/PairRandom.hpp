#ifndef PAIR_RANDOM_ROLLOUT_HEADER_FILE
#define PAIR_RANDOM_ROLLOUT_HEADER_FILE

#include <random>
#include <Types.hpp>
#include <Seeder.hpp>
#include <iostream>

template <typename Env>
class PairRandom {
    public:
        PairRandom();
        double continueRollout(Env& env);
    private:
        std::vector<size_t> arms_;
        std::mt19937 engine_;
};

template <typename Env>
PairRandom<Env>::PairRandom() : engine_(Seeder::getSeed()) {}

template <typename Env>
double PairRandom<Env>::continueRollout(Env& env) {
    int obs;
    double reward = 0;
    bool terminal;

    // get budgets, to evenly spread out arm pull
    auto [b1, b2] = env.getBudgets();
    auto [s1, s2] = env.getSteps();
    b1 -= s1;
    b2 -= s2;

    while (b1 > 0 && b2 > 0) {
        std::bernoulli_distribution dist(b2 / static_cast<double>(b1 + b2));
        size_t action = dist(engine_);

        std::tie(obs, reward, terminal) = env.step(action);
        if (action) --b2;
        else --b1;
    } 

    auto b = std::max(b1, b2);

    while (b > 0) {
        // action should be 0, since there is no pull budget left
        std::tie(obs, reward, terminal) = env.step(0);
        --b;
    }

    return reward;
}

#endif

