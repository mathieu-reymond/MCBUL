#ifndef MOROBIN_HEADER_FILE
#define MOROBIN_HEADER_FILE

#include <random>
#include <Types.hpp>
#include <Seeder.hpp>
#include <iostream>

template <typename Env>
class MOROBIN {
    public:
        MOROBIN();
        double continueRollout(Env& env);
    private:
        std::vector<size_t> arms_;
        std::mt19937 engine_;
};

template <typename Env>
MOROBIN<Env>::MOROBIN() : engine_(Seeder::getSeed()) {}

template <typename Env>
double MOROBIN<Env>::continueRollout(Env& env) {
    int obs;
    double reward = 0;
    bool terminal;

    // get budgets, to evenly spread out arm pull
    auto [b1, b2] = env.getBudgets();
    auto [s1, s2] = env.getSteps();
    b1 -= s1;
    b2 -= s2;

    // interested in number of arms
    if (b1) {
        const size_t A = env.getBandit2Posterior().getA();
        arms_.resize(A);
        std::iota(std::begin(arms_), std::end(arms_), 0);

        std::shuffle(std::begin(arms_), std::end(arms_), engine_);

        size_t index = 0;
        while (b1--) {
            std::tie(obs, reward, terminal) = env.step(arms_[index++]);

            index = index % A;
        }
    }
    // use query budget at the end
    while (b2--) {
        // action should be 0, since there is no pull budget left
        std::tie(obs, reward, terminal) = env.step(0);
    }
    return reward;
}

#endif
