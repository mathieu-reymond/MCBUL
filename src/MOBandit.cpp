#include <MOBandit.hpp>

#include <Seeder.hpp>
#include <MOBanditNormalPosterior.hpp>

MOBandit::MOBandit(const std::vector<Vector> & mus, const std::vector<Vector> & stds) :
        rand_(Seeder::getSeed())
{
    assert(mus.size() > 0);
    assert(mus.size() == stds.size());

    arms_.resize(mus.size());

    const size_t A = mus.size();
    const size_t O = mus[0].size();

    for (size_t a = 0; a < A; ++a) {
        assert(static_cast<size_t>(mus[a].size()) == O);
        assert(static_cast<size_t>(stds[a].size()) == O);

        arms_[a].reserve(O);

        for (size_t o = 0; o < O; ++o)
            arms_[a].emplace_back(mus[a][o], stds[a][o]);
    }

    sample_.resize(O);
}

MOBandit::MOBandit(const MOBanditNormalPosterior & bpost, InitType i) :
        arms_(bpost.getA()), sample_(bpost.getW()), rand_(Seeder::getSeed())
{
    reset(bpost, i);
}

void MOBandit::reset(const MOBanditNormalPosterior & bpost, InitType i) {
    assert(bpost.getA() == getA());
    assert(bpost.getW() == getW());

    for (size_t a = 0; a < getA(); ++a)
        reset(a, bpost, i);
}

void MOBandit::reset(size_t a, const MOBanditNormalPosterior & bpost, InitType i) {
    auto & arm = arms_[a];

    arm.reserve(getW());
    arm.clear();

    // Emplace back the new distributions by unpacking the pair as parameters.
    for (size_t o = 0; o < getW(); ++o) {
        if (i == InitType::SAMPLE)
            std::apply([&](auto... params){arm.emplace_back(params...);}, bpost.sampleMeanStd(a, o, rand_));
        else
            std::apply([&](auto... params){arm.emplace_back(params...);}, bpost.maxLikelihoodMeanStd(a, o));
    }
}

const Vector & MOBandit::sampleR(const size_t a) const {
    for (size_t o = 0; o < getW(); ++o)
        sample_[o] = arms_[a][o](rand_);
    return sample_;
}

double MOBandit::getMean(size_t a, size_t o) const {
    return arms_[a][o].mean();
}
double MOBandit::getStd(size_t a, size_t o) const {
    return arms_[a][o].stddev();
}

size_t MOBandit::getA() const { return arms_.size(); }
size_t MOBandit::getW() const { return sample_.size(); }
const std::vector<std::vector<MOBandit::Dist>> & MOBandit::getArms() const { return arms_; }
