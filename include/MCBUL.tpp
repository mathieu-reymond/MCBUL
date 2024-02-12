#include <limits>
#include <fstream>
#include <Seeder.hpp>

template <typename Env, typename RollStrategy>
std::vector<std::tuple<unsigned, double, unsigned>> computeTreeDepths(const MCBUL<Env, RollStrategy> & mopomcp, std::ostringstream * outDot = nullptr);
template <typename Env, typename RollStrategy>
std::tuple<unsigned, double, unsigned, unsigned> computeTreeDepths(const typename MCBUL<Env, RollStrategy>::ObservationNode & node, std::ostringstream * outDot);
template <typename Env, typename RollStrategy>
std::tuple<unsigned, double, unsigned, unsigned> computeTreeDepths(const typename MCBUL<Env, RollStrategy>::ActionNode & node, std::ostringstream * outDot);

template <typename Env, typename RollStrategy>
MCBUL<Env, RollStrategy>::ActionNode::ActionNode(size_t a, size_t W) :
        action(a), V(0.0), N(0), children(W) {}

template <typename Env, typename RollStrategy>
MCBUL<Env, RollStrategy>::MCBUL(double c, double k0, double a0) : c_(c), k0_(k0), a0_(a0), engine_(Seeder::getSeed()) {}

template <typename Env, typename RollStrategy>
const typename MCBUL<Env, RollStrategy>::ObservationNode & MCBUL<Env, RollStrategy>::getRoot() const {
    return root_;
}

template <typename Env, typename RollStrategy>
size_t getUCB(const typename MCBUL<Env, RollStrategy>::ObservationNode* oNode, const double c) {
    size_t maxAction = 0;
    double maxUCB = -std::numeric_limits<double>::infinity();
    const double logStateVisits = std::log(oNode->N+1);
    for (const auto& child : oNode->children) {
        const double ucb = child.V + c*std::sqrt(logStateVisits/child.N);
        if (ucb > maxUCB) {
            maxUCB = ucb;
            maxAction = child.action;
        }
    }
    return maxAction;
}

// ################## STANDARD ROLLOUTS #####################

template <typename Env, typename RollStrategy>
void MCBUL<Env, RollStrategy>::doRollout(Env& env) {
    env.reset();
    path_.clear();

    ObservationNode* oNode = &root_;

    int obs;
    double reward;
    bool terminal;

    // descend the tree until leaf or until done.
    while (true) {
        ++oNode->N;

        // Choose action.
        // NOTE: If we do progressive widening, here we might simply do a
        // push_back of the new ActionNode we need inside the children.
        const size_t actionId = getUCB<Env, RollStrategy>(oNode, c_);

        auto & aNode = oNode->children[actionId];
        ++aNode.N;
        const size_t action = aNode.action;

        path_.push_back(&aNode);

        std::tie(obs, reward, terminal) = env.step(action);
        if (terminal) break;

        if (!aNode.children[obs]) {
            aNode.children[obs] = std::make_unique<ObservationNode>();
            oNode = aNode.children[obs].get();
            break;
        }
        oNode = aNode.children[obs].get();
        // Only get memory for observation node if we are actually going in it.
        if (oNode->children.size() == 0) {
            const size_t currA = env.getCurrentA();
            oNode->children.reserve(currA);
            // NOTE: For progressive widening here you'd only pick the best
            //       two (yeah two!) actions to start with.
            for (size_t a = 0; a < currA; ++a)
                oNode->children.emplace_back(a, env.getCurrentW(a));
        }
    }

    // update newly encountered observation count
    if (!terminal) {
        ++oNode->N;

        // Final walk.
        reward = strategy_.continueRollout(env);
    }

    // Update backwards, from final node to root-node, using the reward
    for (auto anp : path_)
        anp->V += (reward - anp->V) / anp->N;
}

// ################## OBS PROGRESSIVE WIDENING ROLLOUTS #####################

template <typename Env, typename RollStrategy>
void MCBUL<Env, RollStrategy>::doRolloutOPW(Env& env) {
    env.reset();
    path_.clear();

    ObservationNode* oNode = &root_;

    int obs;
    double reward;
    bool terminal;

    // descend the tree until leaf or until done.
    while (true) {
        ++oNode->N;

        // Choose action.
        // NOTE: If we do progressive widening, here we might simply do a
        // push_back of the new ActionNode we need inside the children.
        const size_t actionId = getUCB<Env, RollStrategy>(oNode, c_);

        auto & aNode = oNode->children[actionId];
        ++aNode.N;
        const size_t action = aNode.action;

        path_.push_back(&aNode);
        // std::cout << "pushed initial node "
        //           << "children size:  "
        //           << aNode.children.size()
        //           << ", N: " << aNode.N
        //           << ", pw metric: "
        //           << k0_ * std::pow(aNode.N, a0_)
        //           << "with k0_ " << k0_
        //           << " and a0_ " << a0_
        //           << "\n";

        if (aNode.children.size() > k0_ * std::pow(aNode.N, a0_)) {
            // std::cout << "going for an existing child " << "\n";
            // Randomly pick one of the existing observation nodes
            std::uniform_int_distribution<size_t> sampleObs(0, aNode.children.size() - 1);
            // Advance tree
            oNode = aNode.children[sampleObs(engine_)].get();
            // Do a stepLoad from it.
            std::tie(obs, reward, terminal) = env.stepLoad(action, oNode->data);
        } else {
            // std::cout << "creating a new child" << "\n";
            // Create new observation node
            aNode.children.emplace_back(std::make_unique<ObservationNode>());
            // Advance tree
            oNode = aNode.children.back().get();
            // std::cout << "take the newly created child" << "\n";
            // Do a stepStore from it
            std::tie(obs, reward, terminal) = env.stepStore(action, oNode->data);
            // std::cout << "did an env step" << "\n";
            // Allocate children actions if needed
            if (!terminal) {
                const size_t currA = env.getCurrentA();
                oNode->children.reserve(currA);
                // NOTE: For progressive widening here you'd only pick the best
                //       two (yeah two!) actions to start with.
                for (size_t a = 0; a < currA; ++a)
                    oNode->children.emplace_back(a, 0);
            }
            // Added leaf, go to rollout.
            break;
        }

        if (terminal) break;
    }

    // update newly encountered observation count
    if (!terminal) {
        ++oNode->N;

        // Final walk.
        reward = strategy_.continueRollout(env);
    }

    // Update backwards, from final node to root-node, using the reward
    for (auto anp : path_)
        anp->V += (reward - anp->V) / anp->N;
}

template <typename Env, typename RollStrategy>
size_t MCBUL<Env, RollStrategy>::recommendAction(Env& env, unsigned nRollouts) {
    // !! initial reset of env to get the correct number of steps
    env.reset();
    const size_t nActions = env.getStartA();

    // Reset tree.
    root_ = ObservationNode();
    root_.children.reserve(nActions);
    for (size_t a = 0; a < nActions; ++a)
        // ! SWITCH TO DO PW ROLLOUTS
        root_.children.emplace_back(a, env.getCurrentW(a));
        // root_.children.emplace_back(a, 0);

    // Reserve enough nodes in the path vector so we don't have to allocate.
    auto [b1, b2] = env.getBudgets();
    path_.reserve(b1 + b2);

    // Do the rollouts
    for (unsigned i = 0; i < nRollouts; ++i) {
        // if (!(i % (nRollouts/100))) {
        //     if (!(i % (nRollouts/10))) std::cout << i;
        //     else std::cout << '.';
        // }
        // std::cout << std::flush;

        // SWITCH TO DO PW ROLLOUTS
        doRollout(env);
        // doRolloutOPW(env);
    }
    // std::cout << '\n';

    //for (size_t aId = 0; aId < nActions; ++aId) {
    //    const auto & aNode = root_.children[aId];
    //    const auto a = aNode.action;
    //    std::cout << child.N << " (" << child.getUCB(root_.getVisits(), 0.) << ") ";
    //}
    //std::cout << '\n';
    // using c=0 means only average reward is taken into account in UCB

    // std::ostringstream graph;
    // const auto depths = computeTreeDepths(*this); //, &graph);

    // std::ofstream file("/tmp/graphxxx");
    // file << graph.str();

    // std::cout << "TREE DEPTHS:\n";
    // for (auto [min, avg, max] : depths) {
    //    std::cout << "- MIN: " << min << "; AVG: " << avg << "; MAX: " << max << '\n';
    // }

    // for (const auto& rootChild : root_.children) {
    //     std::cout << "==CHILD==\n";
    //     for (const auto& aChild : rootChild.children) {
    //         std::cout << aChild->data.diffOrR.transpose() << " " << aChild->N << "\n";
    //     }
    // }

    // std::cout << "UCB, recommend: ";
    // std::cout << "Root children values (and visits): ";
    // for (const auto& child : root_.children) std::cout << child.V << " (" << child.N << ") ";
    // std::cout << '\n';
    size_t action = getUCB<Env, RollStrategy>(&root_, 0);
    return action;
}

#include <TreeDepths.tpp>
