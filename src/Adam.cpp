#include <Adam.hpp>

Adam::Adam(Vector * point, const Vector & gradient, const Float alpha, const Float beta1, const Float beta2, const Float epsilon) :
    point_(point), gradient_(&gradient),
    m_(point_->size()), v_(point_->size()),
    beta1_(beta1), beta2_(beta2), alpha_(alpha), epsilon_(epsilon),
    step_(1)
{
    reset();
}

void Adam::step() {
    assert(point_);
    assert(gradient_);

    m_ = beta1_ * m_ + (1.0 - beta1_) * (*gradient_);
    v_ = beta2_ * v_ + (1.0 - beta2_) * (*gradient_).array().square().matrix();

    const Float alphaHat = alpha_ * std::sqrt(1.0 - std::pow(beta2_, step_)) / (1.0 - std::pow(beta1_, step_));

    (*point_).array() -= alphaHat * m_.array() / (v_.array().sqrt() + epsilon_);

    ++step_;
}

void Adam::reset() {
    m_.fill(0.0);
    v_.fill(0.0);
    step_ = 1;
}

void Adam::reset(Vector * point, const Vector & gradient) {
    point_ = point;
    gradient_ = &gradient;
    reset();
}

void Adam::setBeta1(Float beta1) { beta1_ = beta1; }
void Adam::setBeta2(Float beta2) { beta2_ = beta2; }
void Adam::setAlpha(Float alpha) { alpha_ = alpha; }
void Adam::setEpsilon(Float epsilon) { epsilon_ = epsilon; }

Float Adam::getBeta1() const { return beta1_; }
Float Adam::getBeta2() const { return beta2_; }
Float Adam::getAlpha() const { return alpha_; }
Float Adam::getEpsilon() const { return epsilon_; }
