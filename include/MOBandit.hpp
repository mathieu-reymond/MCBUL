#ifndef MO_BANDIT_HEADER_FILE
#define MO_BANDIT_HEADER_FILE

#include <Types.hpp>

class MOBanditNormalPosterior;

class MOBandit {
    public:
        using Dist = std::normal_distribution<Float>;

        MOBandit(const std::vector<Vector> & mus, const std::vector<Vector> & stds);
        MOBandit(const MOBanditNormalPosterior & posterior, InitType);

        void reset(const MOBanditNormalPosterior & posterior, InitType);
        void reset(size_t arm, const MOBanditNormalPosterior & posterior, InitType);

        const Vector & sampleR(size_t a) const;

        double getMean(size_t a, size_t o) const;
        double getStd(size_t a, size_t o) const;

        size_t getA() const;
        size_t getW() const;
        const std::vector<std::vector<Dist>> & getArms() const;

    private:
        mutable std::vector<std::vector<Dist>> arms_;
        mutable Vector sample_;

        mutable std::mt19937 rand_;
};

/// This is the max absolute difference for which two values can be considered equal.
constexpr auto equalToleranceSmall = 0.000001;
/// This is a relative term used in the checkEqualGeneral functions, where
/// two values may be considered equal if they are within this percentage
/// of each other.
constexpr auto equalToleranceGeneral = 0.00000000001;

inline bool dominates(const Vector & lhs, const Vector & rhs) {
    return (lhs.array() - rhs.array() >= -equalToleranceSmall).minCoeff() ||
           (lhs.array() - rhs.array() >= -lhs.array().min(rhs.array()) * equalToleranceGeneral).minCoeff();
};

// This unfortunately loses the relative check but shouldn't be a huge deal for now.
inline bool dominated(const Vector & diff) {
    return (diff.array() >= -equalToleranceSmall).minCoeff() ||
           (diff.array() <= equalToleranceSmall).minCoeff();
};

#endif
