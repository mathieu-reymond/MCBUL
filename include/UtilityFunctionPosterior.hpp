#ifndef UTILITY_FUNCTION_POSTERIOR_HEADER_FILE
#define UTILITY_FUNCTION_POSTERIOR_HEADER_FILE

#include <StorageEigen.hpp>
#include <BayesianLogisticRegression.hpp>

class UtilityFunction;
class MOBanditNormalPosterior;

class UtilityFunctionPosterior {
    public:
        UtilityFunctionPosterior(size_t W, double multiplier);

        UtilityFunctionPosterior(const UtilityFunctionPosterior &);
        UtilityFunctionPosterior & operator=(const UtilityFunctionPosterior & other);

        void record(const Vector & diff, bool answer);
        void record(const Vector & diff, const UtilityFunction & uf);

        void recordDiffAndPosterior(const Vector & diff, bool answer, const Vector & mean, const Eigen::LLT<Matrix2D> covLLT);

        void optimize(bool restart = true);
        void optimizeMean(bool restart = true);

        const Eigen::Ref<Matrix2D> & getData() const { return data_.matrix; }
        const Eigen::Ref<Vector> & getTarget() const { return target_.vector; }

        double getLeftBoundAngle() const;
        double getRightBoundAngle() const;

        const Vector & getMean() const;
        const Matrix2D & getCov() const;
        const Eigen::LLT<Matrix2D> & getCovLLT() const;
        const BayesianLogisticRegression & getBLR() const;

        size_t getW() const;

    private:
        StorageMatrix2D data_;
        StorageVector target_;

        double leftBoundAngle_, rightBoundAngle_;

        Vector mean_;
        Eigen::LLT<Matrix2D> covLLT_;
        BayesianLogisticRegression blr_;
};

const Vector & suggestQuery2D(const MOBanditNormalPosterior & bpost, const UtilityFunctionPosterior & upost);
const Vector & suggestQueryThompson(const MOBanditNormalPosterior & bpost, const UtilityFunctionPosterior & upost);

#endif
