#ifndef MCBUL_HEADER_FILE
#define MCBUL_HEADER_FILE

#include <vector>
#include <memory>
#include <random>

template <typename Env, typename RollStrategy>
class MCBUL {
    public:
        struct ObservationNode;

        struct ActionNode {
            using Parent = MCBUL<Env, RollStrategy>;

            ActionNode(size_t a, size_t W);

            size_t action;
            double V;
            unsigned N;

            std::vector<std::unique_ptr<ObservationNode>> children;
        };

        struct ObservationNode {
            using Parent = MCBUL<Env, RollStrategy>;

            unsigned N = 0;

            typename Env::Data data;

            std::vector<ActionNode> children;
        };

        MCBUL(double c, double k0, double a0);

        void doRollout(Env& env);
        void doRolloutOPW(Env& env);

        size_t recommendAction(Env& env, unsigned nRollouts);

        const ObservationNode & getRoot() const;

    private:
        // UCT parameter.
        double c_;
        // OBS progressive widening params.
        double k0_, a0_;

        ObservationNode root_;

        std::vector<ActionNode*> path_;
        std::mt19937 engine_;

        RollStrategy strategy_;
};

#include <MCBUL.tpp>

#endif
