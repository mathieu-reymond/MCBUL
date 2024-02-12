#include <cstdlib>
//#include <csignal>
#include <filesystem>
#include <exception>
#include <fstream>
#include <limits>
#include <argparse.hpp>
#include <MOBandit.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <UtilityFunctionParticlePosterior.hpp>
#include <UtilityFunctionPosterior.hpp>
#include <UtilityFunction.hpp>
#include <Environment.hpp>
#include <PairParticleEnvironment.hpp>
#include <ParticleEnvironment.hpp>
#include <MCBUL.hpp>
#include <MOROBIN.hpp>
#include <MOTTTS.hpp>
#include <PairRandom.hpp>
#include <PairInterleaved.hpp>
#include <IO.hpp>
#include <Statistics.hpp>
#include <GitHash.hpp>

void initParticles(Matrix2D & particles) {
    std::mt19937 globalEngine(Seeder::getSeed());
    if (false) {
        // RANDOM
        for (auto i = 0; i < particles.rows(); ++i)
            particles.row(i) = makeRandomProbability(particles.cols(), globalEngine);
    } else {
        // UNIFORM
        const double step = 1.0 / (particles.rows() + 1); // We don't include 1/0 & 0/1
        for (auto i = 0; i < particles.rows(); ++i) {
            const double v = step * (i+1);
            particles(i, 0) = 1.0 - v;
            particles(i, 1) = v;
        }
    }
}

double computeRegret(const std::vector<Vector> &mu, const Vector &weights, size_t recommendedArm) {
    // counters sum up to 1, percentages of arm recommendations
    double bestU = -std::numeric_limits<double>::infinity();
    double worstU = std::numeric_limits<double>::infinity();
    double regret = 0;
    for (size_t i = 0; i < mu.size(); ++i) {
        const double u = mu[i].dot(weights);
        if (u > bestU) bestU = u;
        if (u < worstU) worstU = u;
	if (i == recommendedArm) regret -= u;
    }
    return (regret + bestU)/(bestU-worstU);
    // return (regret + bestU);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("MCBUL");

    program.add_argument("--rollouts")
        .default_value(5000)
        .scan<'i', int>()
        .help("number of MCTS rollouts");

    program.add_argument("--bandit-budget")
        .default_value(20)
        .scan<'i', int>()
        .help("bandit pull budget (wo initial pulls)");

    program.add_argument("--oracle-budget")
        .default_value(3)
        .scan<'i', int>()
        .help("oracle queries budget (wo initial queries)");

    program.add_argument("--score-samples")
        .default_value(1000)
        .scan<'i', int>()
        .help("#samples used for MCTS scoring function (confidence of best arm)");

    program.add_argument("--seed")
        .default_value(0)
        .scan<'i', int>()
        .help("random seed");

    program.add_argument("--multiplier")
        .default_value(1)
        .scan<'i', int>()
        .help("BLR gradient multiplier");

    program.add_argument("--c")
        .default_value(1.0)
        .scan<'g', double>()
        .help("UCB exploration factor");

    program.add_argument("--k0")
        .default_value(1.0)
        .scan<'g', double>()
        .help("UCB exploration factor");

    program.add_argument("--a0")
        .default_value(0.4)
        .scan<'g', double>()
        .help("UCB exploration factor");

    program.add_argument("--file")
        .help("Bandit file to read.");
    
    program.add_argument("--experiments")
        .default_value(1)
        .scan<'i', int>()
        .help("number of experiments on this bandit");

    try {
        program.parse_args(argc, argv);    // Example: ./main --color orange
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    auto rollouts = program.get<int>("--rollouts");
    auto banditBudget_ = program.get<int>("--bandit-budget");
    auto oracleBudget_ = program.get<int>("--oracle-budget");
    auto scoreSamples = program.get<int>("--score-samples");
    auto multiplier = program.get<int>("--multiplier");
    auto c = program.get<double>("--c");
    auto k0 = program.get<double>("--k0");
    auto a0 = program.get<double>("--a0");
    auto seed = program.get<int>("--seed");
    auto experiments = program.get<int>("--experiments");
    const auto filename = program.get<std::string>("--file");

// ============================================
    // BANDIT & UF PARAMETERS
    Vector weights;
    std::vector<Vector> mu, sigma;

    if (filename != "") {
        std::ifstream file(filename);
        std::tie(weights, mu, sigma) = parseModelParameters(file);
    } else {
        throw std::invalid_argument("filename should not be empty");
    }

// ============================================

    Seeder::setRootSeed(seed);
    
    for (size_t experiment = 0; experiment < experiments; ++experiment) {
        int banditBudget = banditBudget_;
	int oracleBudget = oracleBudget_;
         
        MOBandit bandit(mu, sigma);
        MOBanditNormalPosterior banditp(bandit);
        Vector lastSampled;
        const size_t nArms = mu.size();

        // initially pull every arm twice
        for (size_t arm_i = 0; arm_i < nArms; ++arm_i) {
            for (size_t i = 0; i < 2; ++i) {
                lastSampled = bandit.sampleR(arm_i);
                banditp.record(arm_i, lastSampled);
            }
        }
        //DECISION MAKER INIT

        UtilityFunction uf(weights, 0.);

        // Particle filtering initialization
        std::mt19937 eng(Seeder::getSeed());
        [[maybe_unused]] const double noise = 0.05;
        Matrix2D particles(100, bandit.getW());
        initParticles(particles);

        #ifndef USEPARTICLES
            #define USEPARTICLES 1
        #endif

        #if USEPARTICLES == 1
            UtilityFunctionParticlePosterior ufp(particles, noise);
            //MOPairParticleEnv env(banditp, ufp, banditBudget, oracleBudget, scoreSamples);
            //MCBUL<MOPairParticleEnv, PairInterleaved<MOPairParticleEnv>> mopomcp(c, k0, a0);
            MOBanditParticleEnv env(banditp, ufp, banditBudget, oracleBudget, scoreSamples);
            MCBUL<MOBanditParticleEnv, MOTTTS<MOBanditParticleEnv>> mopomcp(c, k0, a0);
        #else
            UtilityFunctionPosterior ufp(bandit.getW(), multiplier);
            MOBanditEnv env(banditp, ufp, banditBudget, oracleBudget, scoreSamples);
            MCBUL<MOBanditEnv, MOROBIN<MOBanditEnv>> mopomcp(c, k0, a0);
        #endif

        for (int i = 0; i < 2; ++i) {
            // 2 initial pulls according to query selector
            ufp.record(suggestQueryThompson(banditp, ufp), uf);
            #if USEPARTICLES == 0
                ufp.optimize();
            #endif

        }
        // std::cout << "initial blr belief: " << ufp.getBLR().getWeights().transpose() << '\n';
        // std::cout << "initial belief:\nW " << ufp.getMean().transpose() << "\n";

        std::vector<unsigned> pulls(mu.size(), 0);

        while (banditBudget > 0 || oracleBudget > 0) {
            // std::cout << "Budget left: " << banditBudget << ", " << oracleBudget << "\n";
            env.setBudgets(banditBudget, oracleBudget);
            // std::cout << "set budgets " << banditBudget << ", " << oracleBudget << std::endl;

            const clock_t start_time = clock();
            unsigned action;
            bool isPull = true;

            if constexpr (std::is_same_v<decltype(env), MOPairParticleEnv>) {
                if (oracleBudget > 0)
                    action = mopomcp.recommendAction(env, rollouts);
                else
                    action = 0;
                if (action == 0 && banditBudget > 0)
                    action = recommendPull(banditp, ufp.getMean(), 0.5, eng);
                else
                    isPull = false;
            } else {
                action = mopomcp.recommendAction(env, rollouts);
                isPull = (banditBudget > 0) && (action < nArms);
            }

            // check if action is pull or query
            if (isPull) {
                // pulling recommended arm
                lastSampled = bandit.sampleR(action);
                banditp.record(action, lastSampled);
                // update corresponding budget
                banditBudget--;
                // log pull
                pulls[action]++;
            } else {
                ufp.record(suggestQueryThompson(banditp, ufp), uf);
                #if USEPARTICLES == 0
                    ufp.optimize();
                #endif

                // update corresponding budget
                oracleBudget--;

                // std::cout << "current belief: W " << ufp.getMean().transpose() << "\n";
                // std::cout << " Cov " << ufp.getCov() << "\n";
            }

            double seconds = float(clock()-start_time) / CLOCKS_PER_SEC;
            // std::cout << "Pulling? " << isPull << ", action " << action << " (" << seconds << "s)\n";
        }

        const auto& ufpMean = ufp.getMean();
        const auto& sampleMu = banditp.getMeans();
        size_t bestArm = 0;
        double bestU = -std::numeric_limits<double>::infinity(); 
        for (size_t currentArm = 0; currentArm < nArms; ++currentArm) {
            double u = sampleMu.row(currentArm).dot(ufpMean);
            if (u > bestU) {
                bestU = u;
                bestArm = currentArm;
            }
        }
        double regret = computeRegret(mu, weights, bestArm);
        std::cout << "recommending: " << bestArm << std::endl;
        std::cout << "regret " << regret << std::endl;
    }

    // Final decision
    // std::cout << "===TRUE ARM VALUES===\n";
    // for (const auto & m : mu)
    //     std::cout << m.dot(weights) << ' ';
    // std::cout << "\n===ARM VALUES===\n";
    // std::cout << (banditp.getMeans()*weights).transpose();
    // std::cout << "\n===ARM EST VALUES===\n";
    // std::cout << (banditp.getMeans()*ufp.getMean()).transpose();

    // // const auto& sampleMu = banditp.getMeans();
    // const auto& sampleSS = banditp.getSampleSquares();
    // for(size_t arm_i = 0; arm_i < nArms; ++arm_i) {
    //     std::cout << "\n===ARM===\n";
    //     for(size_t obj_i = 0; obj_i < static_cast<size_t>(sampleMu.cols()); ++obj_i) {
    //         std::cout << "  " << sampleMu(arm_i, obj_i) << " +- " << sampleSS(arm_i, obj_i)/banditp.getCounts()[arm_i] << "\n";
    //     }
    // }
    // std::cout << "===PULLS ";
    // for (auto& i : pulls) std::cout << i << " ";
    // std::cout << "\n";
    // std::cout << "===ORACLE BELIEF===\n";
    // std::cout << "W: \n" << ufp.getMean().transpose() << std::endl;
    // std::cout << " Cov: \n" << ufp.getCov() << "\n";

    // std::cout << "===FULL DATA===\n ";
    // for (auto& moarm : arms) {
    //     std::cout << "===ARM===\n";
    //     for (auto& arm : moarm.objectiveArms_) {
    //         std::cout << "==OBJ==\n";
    //         for (auto v : arm.samples_)
    //             std::cout << v << ' ';
    //         std::cout << '\n';
    //     }
    // }
    // std::cout << "===ORACLE===\n";
    // for (int i = 0; i < dm.getData().rows(); ++i) {
    //     std::cout << dm.getData().row(i) << "   ===>  " << dm.getTarget()[i] << '\n';
    // }
}
