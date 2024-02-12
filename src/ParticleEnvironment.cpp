#include <ParticleEnvironment.hpp>

#include <algorithm>
#include <stdexcept>

#include <Statistics.hpp>
#include <Seeder.hpp>

// Samples N times and computes measure of uncertainty of best arm.
double computeScore(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost, unsigned nSamples);
// Compares suggested arm with actual true lvl 2 bandit and returns 0/1 reward.
double computeScore(const MOBandit & b, const UtilityFunction & u, const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior& upost);

constexpr bool usingThompsonSuggestions = true;

MOBanditNormalPosterior& MOBanditParticleEnv::getBandit2Posterior() {return bpost2_;}
UtilityFunctionParticlePosterior& MOBanditParticleEnv::getUtilityFunction2Posterior() {return upost2_;}


MOBanditParticleEnv::MOBanditParticleEnv(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior & upost, unsigned banditBudget, unsigned oracleBudget, unsigned scoreSamplesN) :
        bpost1_(bpost), upost1_(upost),
        b2_(bpost, InitType::SAMPLE),
        uf2_(upost, 0.0, InitType::SAMPLE),
        bpost2_(bpost1_), upost2_(upost1_),
        steps_({0, 0}),
        banditBudget_(banditBudget), oracleBudget_(oracleBudget), scoreSamplesN_(scoreSamplesN) {}

std::tuple<size_t, double, bool> MOBanditParticleEnv::step(size_t action) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (action < bpost1_.getA());

    size_t nextState;
    if (isPullAction) {
        ++steps_.first;

        const auto & sampleR = b2_.sampleR(action);
        bpost2_.record(action, sampleR);

        nextState = Bucketer()(sampleR, bpost2_, action);
    } else {
        ++steps_.second;

        const auto & diff = usingThompsonSuggestions ?
                                    suggestQueryThompson(bpost2_, upost2_) :
                                    throw std::logic_error("suggestQuery2D not implemented for ParticlePosterior");

        const bool answer = uf2_.evaluateDiff(diff);
        upost2_.record(diff, answer);

        nextState = answer;
    }

    // discretize next-state for MCTS
    const bool terminal = isTerminal();
    double reward = 0.0;
    if (terminal) {
        if (scoreSamplesN_ > 1) {
            reward = computeScore(bpost2_, upost2_, scoreSamplesN_);
        } else {
            reward = computeScore(b2_, uf2_, bpost2_, upost2_);
        }
    }
    return {nextState, reward, terminal};
}

std::tuple<size_t, double, bool> MOBanditParticleEnv::stepStore(size_t action, Data & out) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (action < bpost1_.getA());

    if (isPullAction) {
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

    return stepLoad(action, out, true);
}

std::tuple<size_t, double, bool> MOBanditParticleEnv::stepLoad(size_t action, const Data & in, bool stored) {
    // check if this is a pull-action or a query-action
    const bool isPullAction = (banditBudget_ > steps_.first) && (action < bpost1_.getA());

    size_t nextState;
    if (isPullAction) {
        ++steps_.first;

        bpost2_.record(action, in.diffOrR);
        // We re-sample the arm we loaded so it is consistent with our current posterior.
        if (!stored) b2_.reset(action, bpost2_, InitType::SAMPLE);

        // Could cache this as well if we wanted to.
        nextState = Bucketer()(in.diffOrR, bpost2_, action);
    } else {
        ++steps_.second;

        // If we are called from stored we have recorded the data already
        if (!stored) upost2_.record(in.diffOrR, in.answer);
        // Regenerate lvl2 UF after loading previous observation.
        uf2_.reset(upost2_, InitType::SAMPLE);

        nextState = in.answer;
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
    return std::tuple(nextState, reward, terminal);
}

void MOBanditParticleEnv::reset() {
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

size_t MOBanditParticleEnv::getCurrentA() const {
    // get number of allowed actions according to budget left
    size_t currentA = 0;
    if (banditBudget_ > steps_.first) currentA += bpost1_.getA();
    if (oracleBudget_ > steps_.second) ++currentA;
    return currentA;
};

size_t MOBanditParticleEnv::getCurrentW(size_t a) const {
    if (banditBudget_ > steps_.first && a < bpost1_.getA())
        return Bucketer().getW();
    return 2;
}

size_t MOBanditParticleEnv::getStartA() const {
    // get number of allowed actions according to budget left
    size_t currentA = 0;
    if (banditBudget_ > 0) currentA += bpost1_.getA();
    if (oracleBudget_ > 0) ++currentA;
    return currentA;
};

size_t MOBanditParticleEnv::getStartW(size_t a) const {
    if (banditBudget_ > 0 && a < bpost1_.getA())
        return Bucketer().getW();
    return 2;
}

bool MOBanditParticleEnv::isTerminal() const {
    return (steps_.first == banditBudget_) && (steps_.second == oracleBudget_);
};

void MOBanditParticleEnv::setBudgets(unsigned banditBudget, unsigned oracleBudget) {
    banditBudget_ = banditBudget;
    oracleBudget_ = oracleBudget;
};

std::pair<unsigned, unsigned> MOBanditParticleEnv::getBudgets() const {
    return {banditBudget_, oracleBudget_};
}

std::pair<unsigned, unsigned> MOBanditParticleEnv::getSteps() const {
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
    return estBestArm == trueBestArm;
}
