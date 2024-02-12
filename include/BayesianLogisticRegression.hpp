#ifndef BAYESIAN_LOGISTIC_REGRESSION_HEADER_FILE
#define BAYESIAN_LOGISTIC_REGRESSION_HEADER_FILE

#include <Types.hpp>
#include <Adam.hpp>

class BayesianLogisticRegression {
    public:
        BayesianLogisticRegression(const Vector & wPrior, const Matrix2D & pPrior, double multiplier = 1.0);
        BayesianLogisticRegression(const BayesianLogisticRegression &) = delete;

        void optimize(const Matrix2D& data, const Vector& results, const Vector & startingPoint);
        void optimizeMean(const Matrix2D& data, const Vector& results, const Vector & startingPoint);

        const Vector & getWeights() const;
        const Matrix2D & getCov() const;

        void setMultiplier(double multiplier);
        double getMultiplier() const;

        void setWeights(const Vector & weights);
        void setCov(const Matrix2D & cov);

        void setPriorWeights(const Vector & wPrior);
        void setPriorPrecision(const Matrix2D & pPrior);

        const Vector & getPriorWeights() const;
        const Matrix2D & getPriorPrecision() const;

        Adam & getAdam();
        const Adam & getAdam() const;

    private:
        double updateGradients(const Matrix2D& data, const Vector& results);

        double multiplier_;

        Vector w0_;
        Matrix2D p0_;

        Vector weights_, gradients_;
        Matrix2D cov_;

        Vector values_;
        Adam adam_;
};

#endif
