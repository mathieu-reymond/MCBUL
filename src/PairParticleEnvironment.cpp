#include <PairParticleEnvironment.hpp>

#include <algorithm>
#include <stdexcept>

#include <Statistics.hpp>
#include <Seeder.hpp>
#include <MOTTTS.hpp>

// Samples N times and computes measure of uncertainty of best arm.
double computeScore(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost, unsigned nSamples);
// Compares suggested arm with actual true lvl 2 bandit and returns 0/1 reward.
double computeScore(const MOBandit & b, const UtilityFunction & u, const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost);

constexpr bool usingThompsonSuggestions = true;

MOBanditNormalPosterior& MOPairParticleEnv::getBandit2Posterior() {return bpost2_;}
UtilityFunctionParticlePosterior& MOPairParticleEnv::getUtilityFunction2Posterior() {return upost2_;}

size_t getObs(MOBanditNormalPosterior& bpost, UtilityFunctionParticlePosterior& upost) {
    double bestU = -std::numeric_limits<double>::infinity();
    double secondU = -std::numeric_limits<double>::infinity();
    size_t bestArm = 0;
    size_t secondArm = 0;
    const auto & weights = upost.getMean();

    for (size_t a = 0; a < bpost.getA(); ++a) {
        const auto & armMean = bpost.getMeans().row(a);
        const double utility = armMean.dot(weights);

        if (utility > bestU) {
            secondU = bestU;
            secondArm = bestArm;
            bestU = utility;
            bestArm = a;
        } else if (utility > secondU) {
            secondU = utility;
            secondArm = a;
        }
    }
    return bestArm*bpost.getA() + secondArm;
}


size_t MOPairParticleEnv::getCurrentW(size_t a) const {
    // get number of children observations for this env
    return bpost2_.getA()*bpost2_.getA();
};

MOPairParticleEnv::MOPairParticleEnv(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior & upost, unsigned banditBudget, unsigned oracleBudget, unsigned scoreSamplesN) :
        rand_(Seeder::getSeed()),
        bpost1_(bpost), upost1_(upost),
        b2_(bpost, InitType::SAMPLE),
        uf2_(upost, 0.0, InitType::SAMPLE),
        bpost2_(bpost1_), upost2_(upost1_),
        steps_({0, 0}),
        banditBudget_(banditBudget), oracleBudget_(oracleBudget), scoreSamplesN_(scoreSamplesN) {}

std::tuple<size_t, double, bool> MOPairParticleEnv::step(size_t pullOrUF) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (pullOrUF == 0);

    if (isPullAction) {
        ++steps_.first;

        // Run Thompson Sampling (TODO: try sample)
        const size_t action = recommendPull(bpost2_, upost2_.getMean(), 0.5, rand_);

        const auto & sampleR = b2_.sampleR(action);
        bpost2_.record(action, sampleR);
    } else {
        ++steps_.second;

        const auto & diff = usingThompsonSuggestions ?
                                    suggestQueryThompson(bpost2_, upost2_) :
                                    throw std::logic_error("suggestQuery2D not implemented for ParticlePosterior");

        const bool answer = uf2_.evaluateDiff(diff);
        upost2_.record(diff, answer);
    }

    size_t nextObs = getObs(bpost2_, upost2_);

    // discretize next-state for MCTS
    bool terminal = isTerminal();
    double reward = 0.0;
    if ((!terminal) && ((steps_.first == banditBudget_) || (steps_.second == oracleBudget_))) {
        while (!terminal) {
            std::tie(nextObs, reward, terminal) = step(0);
        }
        return {nextObs, reward, terminal};
    }
    if (terminal) {
        if (scoreSamplesN_ > 1) {
            reward = computeScore(bpost2_, upost2_, scoreSamplesN_);
        } else {
            reward = computeScore(b2_, uf2_, bpost2_, upost2_);
        }
    }
    return {nextObs, reward, terminal};
}

std::tuple<size_t, double, bool> MOPairParticleEnv::stepStore(size_t pullOrUF, Data & out) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (pullOrUF == 0);

    if (isPullAction) {
        // Run Thompson Sampling (TODO: try sample)
        const size_t action = recommendPull(bpost2_, upost2_.getMean(), 0.5, rand_);

        out.arm = action;
        out.diffOrR = b2_.sampleR(action);
    } else {
        if (usingThompsonSuggestions)
            out.diffOrR = suggestQueryThompson(bpost2_, upost2_);
        else
            throw std::logic_error("suggestQuery2D not implemented for ParticlePosterior");
            // out.diffOrR = suggestQuery2D(bpost2_, upost2_);

        out.answer = uf2_.evaluateDiff(out.diffOrR);

        upost2_.record(out.diffOrR, out.answer);

    }

    return stepLoad(pullOrUF, out, true);
}

std::tuple<size_t, double, bool> MOPairParticleEnv::stepLoad(size_t pullOrUF, const Data & in, bool stored) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (pullOrUF == 0);

    if (isPullAction) {
        ++steps_.first;

        bpost2_.record(in.arm, in.diffOrR);
        // We re-sample the arm we loaded so it is consistent with our current posterior.
        if (!stored) b2_.reset(in.arm, bpost2_, InitType::SAMPLE);
    } else {
        ++steps_.second;

        // If we are called from stored we have recorded the data already
        if (!stored) upost2_.record(in.diffOrR, in.answer);
        // Regenerate lvl2 UF after loading previous observation.
        uf2_.reset(upost2_, InitType::SAMPLE);
    }

    // discretize next-state for MCTS
    const bool terminal = isTerminal();
    double reward = 0.0;
    if (terminal) {
        // scoring function is different depending on number of samples
        if (scoreSamplesN_ > 1) {
            reward = computeScore(bpost2_, upost2_, scoreSamplesN_);
        } else {
            reward = computeScore(b2_, uf2_, bpost2_, upost2_);
        }
    }
    return std::tuple(0, reward, terminal);
}

void MOPairParticleEnv::reset() {
    // Reset steps
    steps_ = {0, 0};

    // Sample new lvl 2 bandit
    if (banditBudget_ > 0) {
        b2_.reset(bpost1_, InitType::SAMPLE);
        bpost2_ = bpost1_;
    }

    if (oracleBudget_ > 0) {
        uf2_.reset(upost1_, InitType::SAMPLE);
        upost2_ = upost1_;
    }
};

size_t MOPairParticleEnv::getCurrentA() const {
    // get number of allowed actions according to budget left
    size_t currentA = 0;
    if (banditBudget_ > steps_.first) ++currentA;
    if (oracleBudget_ > steps_.second) ++currentA;
    return currentA;
};

size_t MOPairParticleEnv::getStartA() const {
    // get number of allowed actions according to budget left
    size_t currentA = 0;
    if (banditBudget_ > 0) ++currentA;
    if (oracleBudget_ > 0) ++currentA;
    return currentA;
};

bool MOPairParticleEnv::isTerminal() const {
    return (steps_.first == banditBudget_) && (steps_.second == oracleBudget_);
};

void MOPairParticleEnv::setBudgets(unsigned banditBudget, unsigned oracleBudget) {
    banditBudget_ = banditBudget;
    oracleBudget_ = oracleBudget;
};

std::pair<unsigned, unsigned> MOPairParticleEnv::getBudgets() const {
    return {banditBudget_, oracleBudget_};
}

std::pair<unsigned, unsigned> MOPairParticleEnv::getSteps() const {
    return steps_;
}

// ----------------------------------------------------------------------------------------------------------------

double computeScore(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost, unsigned nSamples) {
    static std::mt19937 rnd(Seeder::getSeed());

    std::vector<unsigned> scores(bpost.getA());
    Vector sampledW(bpost.getW());
    // sampledW = upost.getMean();

    // going to sample each arm a bunch of times
    for (unsigned i = 0; i < nSamples; ++i) {
        // every time, look which arm performed best
        double bestMean = -std::numeric_limits<float>::infinity();

        size_t bestArm = 0;
        for (size_t a = 0; a < bpost.getA(); ++a) {
            size_t sampledWID = sampleProbability(upost.getWeights(), rnd);
            sampledW = upost.getParticles().row(sampledWID);

            double utility = 0.0;
            for (size_t o = 0; o < bpost.getW(); ++o)
                utility += bpost.sampleMean(a, o, rnd) * sampledW[o];

            if (utility > bestMean) {
                bestMean = utility;
                bestArm = a;
            }
        }
        // increase that arm's score
        ++scores[bestArm];
    }
    // check which arm is believed to be the best
    const double score = *max_element(std::begin(scores), std::end(scores));

    // std::cout << "Computed scores: ";
    // for (auto s : scores)
    //     std::cout << s << ' ';
    // std::cout << '\n';

    return score/nSamples;
}

double computeScore(const MOBandit & b, const UtilityFunction & u, const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost) {
    // Compare true optimal arm vs estimated optimal arm and return 1/0 reward.
    double trueBestMean = -std::numeric_limits<double>::infinity();
    double estBestMean = -std::numeric_limits<double>::infinity();
    size_t trueBestArm = 0, estBestArm = 0;

    const auto & trueWeights = u.getWeights();
    const auto & estWeights = upost.getMean();

    // std::cout << "COMPUTE SCORE:\n"
    //           << "LvL2 weights =     " << trueWeights.transpose()
    //           << "LVL2 est weights = " << estWeights.transpose() << '\n';

    for (size_t a = 0; a < b.getA(); ++a) {
        double trueU = 0.0;
        for (size_t o = 0; o < b.getW(); ++o)
            trueU += b.getMean(a, o) * trueWeights[o];

        const auto & estMean = bpost.getMeans().row(a);
        const double estU = estMean.dot(estWeights);

        if (trueU > trueBestMean) {
            trueBestMean = trueU;
            trueBestArm = a;
        }
        if (estU > estBestMean) {
            estBestMean = estU;
            estBestArm = a;
        }
    }
    // std::cout << "True best arm = " << trueBestArm << " with value " << trueBestMean << '\n'
    //           << "Est. best arm = " << estBestArm  << " with value " << estBestMean  << '\n';

    return estBestArm == trueBestArm;
}
