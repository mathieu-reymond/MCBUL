#include <BayesianLogisticRegression.hpp>

#include <iostream>

inline Vector sigmoid(const Vector & w, const Matrix2D & data) {
    // P(Y = 1) = 1 / (1 + e^{-(b0 + b1 * x1 + b2 * x2 + ...)})
    // P(Y = 0) = 1 - P(Y = 1)
    // P(Y) = P(Y = 1)^Y * (1 - P(Y = 1))^(1-Y)
    //      = P(Y = 1)*Y + (1 - P(Y = 1))*(1 - Y)

    constexpr double trunc = 8.0;
    auto z = (data * w).cwiseMax(-trunc).cwiseMin(trunc).array().exp();

    return z / (1.0 + z);
}

BayesianLogisticRegression::BayesianLogisticRegression(const Vector & wPrior, const Matrix2D & pPrior, double multiplier) :
        multiplier_(multiplier),
        w0_(wPrior), p0_(pPrior),
        weights_(w0_.size()), gradients_(w0_.size()),
        adam_(&weights_, gradients_, 0.05)
{
}

double BayesianLogisticRegression::updateGradients(const Matrix2D& data, const Vector& results) {
    values_ = sigmoid(weights_, data);

    // Log posterior (Eq 4.142):
    //
    // log = -0.5 * (w - w0) * S^-1 * (w - w0)  +  sum_data { y_i * ln(sigmoid) + log(1-sigmoid) * (1 - Y_i) } + const

    double logLikelihood = multiplier_ * (values_.array().log().matrix().dot(results) + ((1.0 - values_.array()).log().matrix()).dot((1.0 - results.array()).matrix()))
    //                               Additional part w.r.t. LogisticRegression
    //                     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                           -0.5 * (weights_ - w0_).transpose() * p0_ * (weights_ - w0_);

    // Gradient (Eq. 4.91 + ?):
    //
    // dot(X.T, sigmoid - Y) + H * (w - w0)
    //                                                                                     Additional part w.r.t. LogisticRegression
    //                                                                                             vvvvvvvvvvvvvvvvvvvvvvv
    gradients_ = multiplier_ * ((values_.array() - results.array()).matrix().transpose() * data) + (p0_ * (weights_ - w0_)).transpose();

    return -logLikelihood;
}

void BayesianLogisticRegression::optimizeMean(const Matrix2D& data, const Vector& results, const Vector & startPoint) {
    // We approximate the posterior of the parameters as a gaussian.
    //
    // See 4.5 of "Pattern Recognition and Machine Learning" by Bishop.
    //
    // MEAN
    // ====
    //
    // MAP of the (correct) mean.

    adam_.reset();
    weights_ = startPoint;
    double costNew = updateGradients(data, results);
    double costOld = std::numeric_limits<double>::max();

    unsigned iterations = 0;

    while (costOld - costNew > 1e-8) {
        costOld = costNew;
        adam_.step();
        costNew = updateGradients(data, results);

        ++iterations;
        if (iterations > 10000) {
            std::cout << "WARNING, 10000 ITERATIONS REACHED, STOPPING\n";
            break;
        }
        // TODO: max iterations?
    }
}

void BayesianLogisticRegression::optimize(const Matrix2D& data, const Vector& results, const Vector & startPoint) {
    optimizeMean(data, results, startPoint);

    // COVARIANCE
    // ==========
    //
    // Laplace approximation via second order (Eq. 4.143)
    //
    // vec S = sigmoid .* (1. - sigmoid)
    // H_post = H + (X.T * (S.asDiagonal() * X))

    cov_ = 1.0/(data.rows()-8+1.0) * (p0_ + multiplier_ * (data.transpose() * ((values_.array() * (1.0 - values_.array())).matrix().asDiagonal() * data))).inverse();
    // cov_ = (p0_ + multiplier_ * (data.transpose() * ((values_.array() * (1.0 - values_.array())).matrix().asDiagonal() * data))).inverse();
}

void BayesianLogisticRegression::setMultiplier(double multiplier) { multiplier_ = multiplier; }
void BayesianLogisticRegression::setWeights(const Vector & weights) { weights_ = weights; }
void BayesianLogisticRegression::setCov(const Matrix2D & cov) { cov_ = cov; }
void BayesianLogisticRegression::setPriorWeights(const Vector & wPrior) { w0_ = wPrior; }
void BayesianLogisticRegression::setPriorPrecision(const Matrix2D & pPrior) { p0_ = pPrior; }

double BayesianLogisticRegression::getMultiplier() const { return multiplier_; }
const Vector & BayesianLogisticRegression::getWeights() const { return weights_; }
const Matrix2D & BayesianLogisticRegression::getCov() const { return cov_; }
const Vector & BayesianLogisticRegression::getPriorWeights() const { return w0_; }
const Matrix2D & BayesianLogisticRegression::getPriorPrecision() const { return p0_; }
