#ifndef PARTICLE_ENVIRONMENT_HEADER_FILE
#define PARTICLE_ENVIRONMENT_HEADER_FILE

#include <MOBandit.hpp>
#include <UtilityFunction.hpp>

#include <MOBanditNormalPosterior.hpp>
#include <UtilityFunctionParticlePosterior.hpp>

#include <Bucketing.hpp>

class MOBanditParticleEnv {
    public:
        struct Data {
            Vector diffOrR;
            bool answer;
        };

        MOBanditParticleEnv(const MOBanditNormalPosterior& bpost, const UtilityFunctionParticlePosterior & upost, unsigned banditBudget, unsigned oracleBudget, unsigned scoreSamplesN);

        void setBudgets(unsigned banditBudget, unsigned oracleBudget);
        std::pair<unsigned, unsigned> getBudgets() const;
        std::pair<unsigned, unsigned> getSteps() const;

        // Returns <obs, reward, terminal>
        std::tuple<size_t, double, bool> stepStore(size_t action, Data & out);
        std::tuple<size_t, double, bool> stepLoad(size_t action, const Data & in, bool stored = false);
        std::tuple<size_t, double, bool> step(size_t action);

        void reset();
        bool isTerminal() const;

        size_t getStartA() const;
        size_t getStartW(size_t a) const;

        size_t getCurrentA() const;
        size_t getCurrentW(size_t a) const;

        MOBanditNormalPosterior& getBandit2Posterior();
        UtilityFunctionParticlePosterior& getUtilityFunction2Posterior();

    private:
        using Bucketer = RewardBucketer;

        const MOBanditNormalPosterior& bpost1_;
        const UtilityFunctionParticlePosterior& upost1_;

        MOBandit b2_;
        UtilityFunction uf2_;

        MOBanditNormalPosterior bpost2_;
        UtilityFunctionParticlePosterior upost2_;

        std::pair<unsigned, unsigned> steps_;

        // Parameters
        unsigned banditBudget_;
        unsigned oracleBudget_;
        unsigned scoreSamplesN_;
};

#endif
