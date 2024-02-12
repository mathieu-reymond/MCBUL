#include <UtilityFunctionPosterior.hpp>

#include <iostream>

#include <UtilityFunction.hpp>
#include <Statistics.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <MOBandit.hpp>
#include <Seeder.hpp>

constexpr double pi = 3.141592653589793116;

UtilityFunctionPosterior::UtilityFunctionPosterior(size_t W, double multiplier) :
        data_(10, W), target_(10),
        leftBoundAngle_(pi), rightBoundAngle_(pi/2),
        mean_(W),
        blr_(
            Vector::Constant(W, 1.0 / W),
            Matrix2D::Identity(W, W) * (1.0 / 0.33) * (1.0 / 0.33),
            multiplier
        )
{
    // This is because we have the bounding angles which only work in 2D.
    assert(W == 2);

    // Init simplex constraints.
    Vector query{{1, 0}};
    data_.push_back(query);
    data_.push_back(query * 10);
    data_.push_back(-query);
    data_.push_back(-query * 10);
    target_.push_back(true);
    target_.push_back(true);
    target_.push_back(false);
    target_.push_back(false);
    query[0] = 0.0; query[1] = 1.0;

    data_.push_back(query);
    data_.push_back(query * 10);
    data_.push_back(-query);
    data_.push_back(-query * 10);
    target_.push_back(true);
    target_.push_back(true);
    target_.push_back(false);
    target_.push_back(false);

    Vector::Constant(W, 1.0 / W);

    // Inialize values from priors for Thompson queries
    mean_ = blr_.getPriorWeights();
    covLLT_ = blr_.getPriorPrecision().inverse().llt();
}

UtilityFunctionPosterior::UtilityFunctionPosterior(const UtilityFunctionPosterior & other) :
        data_(other.data_), target_(other.target_),
        leftBoundAngle_(other.leftBoundAngle_), rightBoundAngle_(other.rightBoundAngle_),
        mean_(other.mean_), covLLT_(other.covLLT_), blr_(other.blr_.getPriorWeights(), other.blr_.getPriorPrecision(), other.blr_.getMultiplier())
{
    assert(getW() == other.getW());
}

UtilityFunctionPosterior & UtilityFunctionPosterior::operator=(const UtilityFunctionPosterior & other) {
    assert(getW() == other.getW());

    data_ = other.data_;
    target_ = other.target_;

    leftBoundAngle_ = other.leftBoundAngle_;
    rightBoundAngle_ = other.rightBoundAngle_;

    mean_ = other.mean_;
    covLLT_ = other.covLLT_;

    blr_.setMultiplier(other.blr_.getMultiplier());

    return *this;
}

void UtilityFunctionPosterior::record(const Vector & diff, const UtilityFunction & uf) {
    record(diff, uf.evaluateDiff(diff));
}

void UtilityFunctionPosterior::record(const Vector & diff, bool answer) {
    if (dominated(diff)) return;

    // Normalize diff to 10 for better estimations
    data_.push_back(diff * 10.0 / diff.array().abs().sum());
    data_.push_back(-diff * 10.0 / diff.array().abs().sum());

    target_.push_back(answer);
    target_.push_back(!answer);

    if (getW() == 0) {
        // Update query boundaries.
        double angle = std::atan2(diff[1], diff[0]);
        if (angle < 0) {
            angle += pi;
            answer = !answer;
        }
        // Check if angle is outside our current bounds.
        if (angle <= rightBoundAngle_ || angle >= leftBoundAngle_) {
            // If it is, but agrees with what we have already, nothing to do.
            if ( answer && angle <= rightBoundAngle_) return;
            if (!answer && angle >= leftBoundAngle_)  return;

            // Otherwise something is wrong; so increase our bounds until we can
            // fit this again.
            const double middleAngle = (leftBoundAngle_ + rightBoundAngle_) / 2.0;
            const double diffAngle = middleAngle - rightBoundAngle_;

            constexpr size_t maxIterations = 500;
            size_t it = 0;
            do {
                leftBoundAngle_ = std::min(pi, leftBoundAngle_ + diffAngle);
                rightBoundAngle_ = std::max(pi/2.0, rightBoundAngle_ - diffAngle);
                ++it;
            }
            while (
                (pi > leftBoundAngle_ || rightBoundAngle_ > pi/2) &&
                (angle > leftBoundAngle_ || rightBoundAngle_ > angle)
                // && it < maxIterations
            );
            if (it == maxIterations) {
                std::cout << "BROKEN WHILE!!! EUGENIO FIX THIS!!!\n"
                          << "diff = " << diff.transpose() << '\n'
                          << "answer = " << answer << '\n'
                          << "leftBoundAngle_ = " << leftBoundAngle_ << '\n'
                          << "rightBoundAngle_ = " << rightBoundAngle_ << '\n'
                          << "diffAngle = " << diffAngle << '\n'
                          << "origAngle = " << std::atan2(diff[1], diff[0]) << '\n'
                          << "currAngle = " << angle << '\n'
                          << "middleAngle = " << middleAngle << '\n';
            }
        }

        if (answer) rightBoundAngle_ = angle;
        else        leftBoundAngle_  = angle;
    }
}

void UtilityFunctionPosterior::recordDiffAndPosterior(const Vector & diff, bool answer, const Vector & mean, const Eigen::LLT<Matrix2D> covLLT) {
    data_.push_back(diff * 10.0 / diff.array().abs().sum());
    data_.push_back(-diff * 10.0 / diff.array().abs().sum());

    target_.push_back(answer);
    target_.push_back(!answer);

    mean_ = mean;
    covLLT_ = covLLT;
}

void UtilityFunctionPosterior::optimize(bool restart) {
    if (restart)
        blr_.optimize(getData(), getTarget(), Vector::Constant(getW(), 1.0 / getW()));
    else
        blr_.optimize(getData(), getTarget(), mean_);

    // std::cout << "UFP: optimize(. Current data:\n" << getData() << '\n' << getTarget().transpose() << '\n';

    mean_ = blr_.getWeights().array() / blr_.getWeights().array().abs().sum();
    covLLT_ = blr_.getCov().llt();
}

void UtilityFunctionPosterior::optimizeMean(bool restart) {
    if (restart)
        blr_.optimizeMean(getData(), getTarget(), Vector::Constant(getW(), 1.0 / getW()));
    else
        blr_.optimizeMean(getData(), getTarget(), mean_);

    mean_ = blr_.getWeights().array() / blr_.getWeights().array().abs().sum();
}

double UtilityFunctionPosterior::getLeftBoundAngle() const { return leftBoundAngle_; }
double UtilityFunctionPosterior::getRightBoundAngle() const { return rightBoundAngle_; }

const Vector & UtilityFunctionPosterior::getMean() const { return mean_; }
const Matrix2D & UtilityFunctionPosterior::getCov() const { return blr_.getCov(); }
const Eigen::LLT<Matrix2D> & UtilityFunctionPosterior::getCovLLT() const { return covLLT_; }

size_t UtilityFunctionPosterior::getW() const { return mean_.size(); }
const BayesianLogisticRegression & UtilityFunctionPosterior::getBLR() const { return blr_; }

/// ###################################################

const Vector & suggestQuery2D(const MOBanditNormalPosterior & bp, const UtilityFunctionPosterior & up) {
    constexpr bool print = false;
    assert(bp.getW() == 2);
    assert(up.getW() == 2);

    static Vector helper(2);

    double leftBoundAngle = up.getLeftBoundAngle();
    double rightBoundAngle = up.getRightBoundAngle();

    // For each comparison, we store its angle and the ids of its arms.
    // We only store indeces to avoid allocating a million Vectors.
    struct angleIds {
        double angle;
        size_t i, j;
    };

    // Reserve space.
    std::vector<angleIds> angleIndeces;
    angleIndeces.reserve(bp.getA() * (bp.getA() - 1) / 2);

    // Generate all valid comparisons.
    //
    // We could already filter the ones outside query bounds but then if we
    // don't have enough we'd have to redo all this work. So we only filter
    // against the dominated quadrants.
    for (size_t i = 0; i < bp.getA() - 1; ++i) {
        for (size_t j = i + 1; j < bp.getA(); ++j) {
            if (print) std::cout << "Comparing arm " << i << " (" << bp.getMeans().row(i) << ") with arm "
                                          << j << " (" << bp.getMeans().row(j) << ")\n";

            const auto dX = bp.getMeans()(i, 0) - bp.getMeans()(j, 0);
            const auto dY = bp.getMeans()(i, 1) - bp.getMeans()(j, 1);

            // Compute angle of this comparison
            double angle = std::atan2(dY, dX);
            if (print) std::cout << "- dX = " << dX << "; dY = " << dY << "; angle = " << angle << '\n';

            // Flip bottom 2 quadrants so we can work only with the top ones.
            const bool flip = angle < 0;
            if (flip) angle += pi;
            if (print) std::cout << "- flipped = " << flip << "; angle = " << angle << '\n';

            // Remove arms in '+' quadrant (top-right) since they are not needed
            if (angle <= pi/2.0) {
                if (print) std::cout << "- angle < " << pi/2.0 << ", skipping\n";
                continue;
            }

            // If flip we store the opposite indeces to recompute the final diff.
            if (flip) angleIndeces.emplace_back(angle, j, i);
            else      angleIndeces.emplace_back(angle, i, j);

            if (print) std::cout << "- emplaced: " << angleIndeces.back().angle << "; "
                      << angleIndeces.back().i << "; " << angleIndeces.back().j << '\n';
        }
    }

    // Sort by angle, highest first (just so it's sorted left-to-right for
    // better mental visualization, doesn't really matter).
    std::sort(std::begin(angleIndeces), std::end(angleIndeces), [](const auto & rhs, const auto & lhs) {
            return rhs.angle > lhs.angle;
    });

    if (print) {
        std::cout << "AFTER SORTING WE HAVE: " << angleIndeces.size() << "entries. Leftmost comp angle = "
              << angleIndeces[0].angle << "; rightmost comp angle = " << angleIndeces.back().angle  << '\n';

        for (const auto& ai : angleIndeces)
            std::cout << "angles: " << ai.angle << "\n";
    }

    size_t leftId = 0, rightId = 0;
    do {
        if (print) std::cout << "Filtering pass; leftBoundAngle = " << leftBoundAngle
                  << "; rightBoundAngle = " << rightBoundAngle << '\n';

        // Figure out the border indeces of the comparisons within boundaries.
        bool foundLeft = false, foundRight = false;
        for (size_t i = 0; i < angleIndeces.size(); ++i) {
            // Left is first element we find with angle lower than left bound.
            if (!foundLeft && angleIndeces[i].angle < leftBoundAngle) {
                foundLeft = true;
                leftId = i;
            }
            // Right is last element we find with angle higher than right bound.
            if (angleIndeces[i].angle <= rightBoundAngle)
                break;
            foundRight = true;
            rightId = i;
        }

        if (print) std::cout << "Finished filtering, leftId = " << leftId << " (found? "
                  << foundLeft << "); rightId = " << rightId << " (found? "
                  << foundRight << ")\n";

        // If we have at least a comparison we can ask, our search is over.
        if (foundLeft && foundRight) break;

        // If we already are considering the whole range, we can't really
        // do much here..
        if (leftBoundAngle == pi and rightBoundAngle == pi/2.0) {
            std::cout << "ERROR! NO COMPARISONS BETWEEN MAX BOUND RANGE!\n";
            std::cout << "Available comparisons are:\n";
            for (const auto & ai : angleIndeces)
                std::cout << "Angle: " << ai.angle << " for arms " << ai.i << ", " << ai.j << '\n';

            // Convert middle angle to diff
            const double dX = -0.7071067811865475244008443621048490392; // std::cos(3/4 * pi);
            const double dY =  0.7071067811865475244008443621048490392; // std::sin(3/4 * pi);

            // We compute the magnitude to have a reasonable diff instead of
            // sin/cos of circle of radius 1
            const double avgMagnitude = bp.getMeans().row(0).norm();

            helper[0] = dX * avgMagnitude;
            helper[1] = dY * avgMagnitude;

            // Return first arm mean, and the same arm plus the diff
            return helper;
        }

        // Else, double the boundary range.
        const double middleAngle = (leftBoundAngle + rightBoundAngle) / 2.0;
        const double diffAngle = middleAngle - rightBoundAngle;

        leftBoundAngle = std::min(pi, leftBoundAngle + diffAngle);
        rightBoundAngle = std::max(pi/2.0, rightBoundAngle - diffAngle);
    } while (true);

    const auto middleId = leftId + (rightId - leftId) / 2;
    const auto & middle = angleIndeces[middleId];

    helper = bp.getMeans().row(middle.i) - bp.getMeans().row(middle.j);

    return helper;
}

const Vector & suggestQueryThompson(const MOBanditNormalPosterior & bpost, const UtilityFunctionPosterior & upost) {
    static Vector retval, bestArm, bestArm2, sampledW;
    static std::mt19937 rnd(Seeder::getSeed());

    retval.resize(bpost.getW());
    bestArm.resize(bpost.getW());
    bestArm2.resize(bpost.getW());
    sampledW.resize(bpost.getW());

    double bestArmV, bestArm2V;
    bestArmV = bestArm2V = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < 100; ++i) {
        // Sample UF
        sampleNormalizedMultivariateNormalInline(upost.getMean(), upost.getCovLLT(), sampledW, rnd);

        // Sample each arm, and remember best two.
        for (size_t a = 0; a < bpost.getA(); ++a) {
            for (size_t o = 0; o < bpost.getW(); ++o)
                retval[o] = bpost.sampleMean(a, o, rnd);

            double utility = retval.dot(sampledW);
            if (utility > bestArm2V) {
                if (utility > bestArmV) {
                    bestArm2V = bestArmV;
                    bestArm2 = bestArm;

                    bestArmV = utility;
                    bestArm = retval;
                } else {
                    bestArm2V = utility;
                    bestArm2 = retval;
                }
            }
        }

        // Try again until we have an informative query to ask.
        if (!dominates(bestArm, bestArm2))
            break;
    }

    retval = bestArm - bestArm2;
    return retval;
}
