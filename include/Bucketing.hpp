#ifndef BUCKETING_HEADER_FILE
#define BUCKETING_HEADER_FILE

#include <Types.hpp>

class MOBanditNormalPosterior;

struct RewardBucketer {
    static constexpr unsigned precision = 10;

    size_t getW() const;
    size_t operator()(const Vector & rew, const MOBanditNormalPosterior & bpost, size_t a);
};


#endif
