#ifndef AI_TOOLBOX_ADAM_HEADER_FILE
#define AI_TOOLBOX_ADAM_HEADER_FILE

#include <Types.hpp>

/**
 * @brief This class implements the ADAM gradient descent algorithm.
 *
 * This class keeps things simple and fast. It takes two pointers to two
 * equally-sized vectors; one used to track the currently examined point,
 * and the other to provide Adam with the gradient.
 *
 * This class expects you to compute the gradient of the currently examined
 * point. At each step() call, the point vector is updated following the
 * gradient using the Adam algorithm.
 *
 * We take pointers rather than references so that the pointers can be
 * updated as needed, while the class instance kept around. This only works
 * if the new vectors have the same size as before, but it allows to avoid
 * reallocation of the internal helper vectors.
 */
class Adam {
    public:
        /**
         * @brief Basic constructor.
         *
         * We expect the pointers to not be null, and the vectors to be preallocated.
         *
         * The point vector should contain the point where to start the
         * gradient descent process. The gradient vector should contain
         * the gradient at that point.
         *
         * @param point A pointer to preallocated space where to write the point.
         * @param gradient A pointer to preallocated space containing the current gradient.
         * @param alpha Adam's step size/learning rate.
         * @param beta1 Adam's exponential decay rate for first moment estimates.
         * @param beta2 Adam's exponential decay rate for second moment estimates.
         * @param epsilon Additive parameter to prevent division by zero.
         */
        Adam(Vector * point, const Vector & gradient, Float alpha = 0.001, Float beta1 = 0.9, Float beta2 = 0.999, Float epsilon = 1e-8);

        /**
         * @brief This function updates the point using the currently set gradient.
         *
         * This function overwrites the vector pointed by the `point`
         * pointer, by following the currently set gradient.
         *
         * It is expected that the gradient is correct and has been updated
         * by the user before calling this function.
         */
        void step();

        /**
         * @brief This function resets the gradient descent process.
         *
         * This function clears all internal values so that the gradient
         * descent process can be restarted from scratch.
         *
         * The point vector is not modified.
         */
        void reset();

        /**
         * @brief This function resets the gradient descent process.
         *
         * This function clears all internal values so that the gradient
         * descent process can be restarted from scratch.
         *
         * The point and gradient pointers are updated with the new inputs.
         */
        void reset(Vector * point, const Vector & gradient);

        /**
         * @brief This function sets the current learning rate.
         */
        void setAlpha(Float alpha);

        /**
         * @brief This function sets the current exponential decay rate for first moment estimates.
         */
        void setBeta1(Float beta1);

        /**
         * @brief This function sets the current exponential decay rate for second moment estimates.
         */
        void setBeta2(Float beta2);

        /**
         * @brief This function sets the current additive division parameter.
         */
        void setEpsilon(Float epsilon);

        /**
         * @brief This function returns the current learning rate.
         */
        Float getAlpha() const;

        /**
         * @brief This function returns the current exponential decay rate for first moment estimates.
         */
        Float getBeta1() const;

        /**
         * @brief This function returns the current exponential decay rate for second moment estimates.
         */
        Float getBeta2() const;

        /**
         * @brief This function returns the current additive division parameter.
         */
        Float getEpsilon() const;

    private:
        Vector * point_;
        const Vector * gradient_;
        Vector m_, v_;

        Float beta1_, beta2_, alpha_, epsilon_;
        unsigned step_;
};

#endif
