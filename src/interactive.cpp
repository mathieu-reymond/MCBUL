#include <iostream>
#include <fstream>
#include <IO.hpp>

#include <argparse.hpp>

#include <Seeder.hpp>
#include <Matplotlib.hpp>
#include <DrawUtils.hpp>
#include <Statistics.hpp>

#include <Types.hpp>

#include <MOBandit.hpp>
#include <MOBanditNormalPosterior.hpp>
#include <UtilityFunction.hpp>
#include <UtilityFunctionPosterior.hpp>
#include <UtilityFunctionParticlePosterior.hpp>
#include <BayesianLogisticRegression.hpp>

namespace plt = matplotlibcpp;

// HELPER FUNCTIONS
Float radToDeg(Float rad);
Matrix2D get2DBoundedPointsFromSlope(Float slope, Float minX, Float maxX, Float minY, Float maxY);
void plotBanditPosteriorStats(const MOBanditNormalPosterior & bpost);
void plotBanditStats(const MOBandit & b, const UtilityFunction & uf);
void plotUtilityFunctionPosterior(const UtilityFunctionPosterior & upost);
void draw(const MOBanditNormalPosterior & bpost, const UtilityFunctionPosterior & upost);
void draw(const MOBanditNormalPosterior & bpost, const UtilityFunctionParticlePosterior & upost);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("MOBanditExplorer");

    program.add_argument("--seed")
        .default_value(0)
        .scan<'i', int>()
        .help("random seed");

    program.add_argument("--multiplier")
        .default_value(1)
        .scan<'i', int>()
        .help("BLR gradient multiplier");

    program.add_argument("--file")
        .default_value(std::string(""))
        .help("Bandit file to read.");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    const auto seed = program.get<int>("--seed");
    const auto multiplier = program.get<int>("--multiplier");
    const auto filename = program.get<std::string>("--file");

    std::cout << "####################################################\n"
              << "#### WELCOME TO THE INTERACTIVE BANDIT EXPLORER ####\n"
              << "####################################################\n"
              << "\n"
              << "Current initial parameters:\n"
              << "- Seed = " << seed << '\n'
              << "- BLR multiplier = " << multiplier << '\n';

    if (filename != "")
        std::cout << "- Using FILE: " << filename << '\n';
    else
        std::cout << "- Using inline model.\n";

    Seeder::setRootSeed(seed);

    // BANDIT & UF PARAMETERS
    Vector weights;
    std::vector<Vector> mu, sigma;

    if (filename != "") {
        std::ifstream file(filename);
        std::tie(weights, mu, sigma) = parseModelParameters(file);
    } else {
        mu = {Vector{{2, 0.}}, Vector{{1.2, 0.1}}, Vector{{1, 0.5}}, Vector{{0.9, 0.55}}, Vector{{.3, 0.7}}, Vector{{0, 0.85}}, Vector{{1.1, 0.4}}, Vector{{1.5, 0.05}}};
        sigma = {Vector{{2., 1.}}, Vector{{1., 1.}}, Vector{{1., 3.}}, Vector{{1., 1.}}, Vector{{1., 1.}}, Vector{{2, 1.}}, Vector{{2., 1.}}, Vector{{1., 3.}}};
        for (auto & e : sigma)
            e /= 10;
        weights = Vector{{0.2, 0.8}};
    }

    // BANDIT & UF INITIALIZATION
    MOBandit bandit(mu, sigma);
    MOBanditNormalPosterior bpost(bandit, true);

    UtilityFunction uf(weights, 0.0);
    UtilityFunctionPosterior upost(bandit.getW(), multiplier);

    std::mt19937 eng(Seeder::getSeed());

    Matrix2D particles(100, bandit.getW());
    for (auto i = 0; i < particles.rows(); ++i)
        particles.row(i) = makeRandomProbability(bandit.getW(), eng);
    UtilityFunctionParticlePosterior upostpp(particles, 0.05);

    // BANDIT INITIAL PULLS
    for (size_t i = 0; i < 2; ++i)
        for (size_t a = 0; a < bandit.getA(); ++a)
            bpost.record(a, bandit.sampleR(a));

    // UF INITIAL QUERIES (select auto or manual)
    // Automatic UF queries
    if (true) {
        Vector query(bandit.getW());
        for (size_t i = 0; i < 2; ++i) {
            // query = suggestQuery2D(bpost, upost);
            query = suggestQueryThompson(bpost, upost);
            upost.record(query, uf.evaluateDiff(query));
            upost.optimize();
        }
        for (size_t i = 0; i < 2; ++i) {
            query = suggestQueryThompson(bpost, upostpp);
            std::cout << "PP QUERY: " << query.transpose() << '\n';
            upostpp.record(query, uf);
        }
    }
    // Manual UF queries
    else {
        Matrix2D queries(5, bandit.getW());
        queries << -1.3547882 ,  0.66143703, -0.48726892,  0.14445117, -0.26566756,  0.05441031, -0.89866757,  0.20960783, -0.89866757,  0.20960783;

        for (auto i = 0; i < 2; ++i)
            upost.record(queries.row(i), uf);
    }

    // INITIAL DRAW
    draw(bpost, upostpp);

    // INTERATIVE TERMINAL SETUP
    std::uniform_int_distribution<size_t> dist(0, bandit.getA()-1);
    std::vector<int> actionSequence; // Logs session

    // BEGIN
    while (true) {
        std::cout << "\nSelect an action ([0, " << bandit.getA() - 1 << "] for pulls, [" << bandit.getA() << "] for UF, [a] to print arms, [b] to print belief, [s] to sample beliefs, [t] for true bandit, other to quit):\n";

        size_t actionNum = 0;
        bool isActionSizet = false;

        std::string action;
        std::cin >> action;

        // Check if action is a positive number
        try {
            const int parseInt = std::stoi(action);
            if (parseInt >= 0) {
                actionNum = parseInt;
                isActionSizet = true;
            }
        } catch (...) {}

        // PULL ACTION
        if (isActionSizet && actionNum < bandit.getA()) {
            actionSequence.push_back(actionNum);

            bpost.record(actionNum, bandit.sampleR(actionNum));

            draw(bpost, upostpp);
        // UF QUERY ACTION
        } else if (isActionSizet && actionNum == bandit.getA()) {
            actionSequence.push_back(actionNum);

            // const auto & diff = suggestQuery2D(bpost, upost);
            const auto & diff = suggestQueryThompson(bpost, upost);
            std::cout << "Querying for diff " << diff.transpose() << '\n';

            upost.record(diff, uf);
            upost.optimize(false);

            { auto query = suggestQueryThompson(bpost, upostpp);
            std::cout << "PP QUERY: " << query.transpose() << '\n';
            upostpp.record(query, uf); }

            draw(bpost, upost);
        // OTHERS
        } else if (action == "a") {
            // plotBanditPosteriorStats(bpost);
            draw(bpost, upostpp);
        } else if (action == "b") {
            plotUtilityFunctionPosterior(upost);
        } else if (action == "t") {
            plotBanditStats(bandit, uf);
        } else if (action == "s") {
            Vector sample = upost.getMean();
            for (size_t i = 0; i < 10; ++i) {
                sampleNormalizedMultivariateNormalInline(upost.getMean(), upost.getCovLLT(), sample, eng);
                std::cout << sample.transpose() << '\n';
            }
        // QUITS
        } else
            break;
    }

    std::cout << "#################\n"
              << "#### GOODBYE ####\n"
              << "#################\n"
              << '\n'
              << "Information to reproduce this session:\n"
              << "- Seed = " << seed << '\n'
              << "- BLR multiplier = " << multiplier << '\n';
    if (actionSequence.size()) {
        std::cout << "- Actions performed in this session:\n";
        for (auto a : actionSequence)
            std::cout << a << ' ';
        std::cout << '\n';
    } else {
        std::cout << "- No actions were performed in this session\n";
    }
}

// ##############################################################################################################################
// ##############################################################################################################################
// ##############################################################################################################################

Float radToDeg(Float rad) {
    return rad * (180.0/3.141592653589793238463);
}

Matrix2D get2DBoundedPointsFromSlope(Float slope, Float minX, Float maxX, Float minY, Float maxY) {
    Matrix2D coords(2, 2);
    coords(0,0) = minX;
    coords(0,1) = slope * minX;
    coords(1,0) = maxX;
    coords(1,1) = slope * maxX;

    if (coords(0,1) > maxY) {
        coords(0,1) = maxY;
        coords(0,0) = maxY / slope;
    } else if (coords(0,1) < minY) {
        coords(0,1) = minY;
        coords(0,0) = minY / slope;
    }

    if (coords(1,1) > maxY) {
        coords(1,1) = maxY;
        coords(1,0) = maxX / slope;
    } else if (coords(1,1) < minY) {
        coords(1,1) = minY;
        coords(1,0) = minY / slope;
    }

    return coords;
}

void plotBanditPosteriorStats(const MOBanditNormalPosterior & bpost) {
    const auto & samples = bpost.getSamples();

    for (size_t a = 0; a < bpost.getA(); ++a) {
        std::cout << "## Data for Arm " << a << " ##\n";
        std::cout << samples[a].matrix << '\n';
    }
    std::cout << '\n';
    for (size_t a = 0; a < bpost.getA(); ++a) {
        const auto mean = bpost.getMeans().row(a);
        const auto N = bpost.getCounts()[a];

        std::cout << "Arm " << a << " (sampled: " << N << "):\n";
        std::cout << "- Mean: " << mean << '\n';
        std::cout << "- SS: " << bpost.getSampleSquares().row(a) << '\n';
        std::cout << "- STD: ";
        for (size_t o = 0; o < bpost.getW(); ++o)
            std::cout << bpost.maxLikelihoodMeanStd(a, o).second << ' ';
        std::cout << '\n';
    }

    plt::clf();
    for (size_t a = 0; a < bpost.getA(); ++a) {
        const auto mean = bpost.getMeans().row(a);
        const auto N = bpost.getCounts()[a];
        const auto postVar = (bpost.getSampleSquares().row(a) / (N * (N-1)));

        auto coords = plotEllipse(mean, postVar.asDiagonal());
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "--"}});

        const auto & data = bpost.getSamples()[a].matrix;

        plt::scatter(toVec(data.col(0)), toVec(data.col(1)), 5.0);
        plt::text(mean[0], mean[1], std::to_string(a));
    }
    plt::pause(0.001);
}

void plotBanditStats(const MOBandit & b, const UtilityFunction & uf) {
    plt::clf();
    Vector mean(b.getW()), std(b.getW());
    for (size_t a = 0; a < b.getA(); ++a) {
        for (size_t o = 0; o < b.getW(); ++o) {
            mean[o] = b.getMean(a, o);
            std[o] = b.getStd(a, o);
        }

        std::cout << "\nArm " << a << ":\n";
        std::cout << "- Mean: " << mean.transpose() << '\n';
        std::cout << "- STD: " << std.transpose();

        auto coords = plotEllipse(mean, std.array().square().matrix().asDiagonal());
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "--"}});

        plt::text(mean[0], mean[1], std::to_string(a));
    }
    std::stringstream title;
    title << "Weights=" << uf.getWeights().transpose();
    plt::title(title.str());

    std::cout << '\n';
    plt::pause(0.001);
}

void plotUtilityFunctionPosterior(const UtilityFunctionPosterior & upost) {
    plt::clf();

    const Vector & w = upost.getMean();
    std::cout << "Current weight is: " << w.transpose() << '\n';
    std::cout << "Current cov is:\n" << upost.getCov() << '\n';
    std::cout << "Current angles (rad) are: [" << upost.getLeftBoundAngle() << ", " << upost.getRightBoundAngle() << "]\n";
    std::cout << "Current angles (deg) are: [" << radToDeg(upost.getLeftBoundAngle()) << ", " << radToDeg(upost.getRightBoundAngle()) << "]\n";
    std::cout << "Comparison data:\n";

    const auto & data = upost.getData();
    const auto & targets = upost.getTarget();

    // Plot datapoints

    Float minX = 1000, maxX = -1000;
    Float minY = 1000, maxY = -1000;
    for (auto i = 0; i < targets.size(); ++i) {
        if (targets[i]) {
            plt::plot(toVecD(data(i, 0)), toVecD(data(i, 1)), {{"marker", "x"}, {"color", "red"}, {"linestyle", " "}});
        } else {
            plt::plot(toVecD(data(i, 0)), toVecD(data(i, 1)), {{"marker", "o"}, {"color", "blue"}, {"linestyle", " "}});
        }
        minX = std::min(minX, data(i, 0));
        maxX = std::max(maxX, data(i, 0));
        minY = std::min(minY, data(i, 1));
        maxY = std::max(maxY, data(i, 1));
        std::cout << targets[i] << " <==> " << data.row(i) << '\n';
    }

    {
        // Normally the slope for w would be w[1] / w[0]; however here we are
        // in diff space so it becomes -w[0] / w[1]
        auto coords = get2DBoundedPointsFromSlope(- w[0] / w[1], minX, maxX, minY, maxY);
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "--"}});
    }

    // Plot angles
    // y = tan(angle) * x
    for (auto angle : {upost.getLeftBoundAngle(), upost.getRightBoundAngle()}) {
        const Float coeff = std::tan(angle);

        auto coords = get2DBoundedPointsFromSlope(coeff, minX, maxX, minY, maxY);
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "dotted"}});
    }

    plt::pause(0.001);
}

std::tuple<Float, Float, Float, Float> draw(const MOBanditNormalPosterior & bpost) {
    Float minX, minY, maxX, maxY;
    minX = minY = std::numeric_limits<Float>::infinity();
    maxX = maxY = -std::numeric_limits<Float>::infinity();

    for (size_t a = 0; a < bpost.getA(); ++a) {
        const auto mean = bpost.getMeans().row(a);
        const auto N = bpost.getCounts()[a];
        const auto postVar = (bpost.getSampleSquares().row(a) / (N * (N-1)));

        auto coords = plotEllipse(mean, postVar.asDiagonal());
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "--"}});

        plt::text(mean[0], mean[1], std::to_string(a));

        minX = std::min(minX, coords.col(0).minCoeff());
        maxX = std::max(maxX, coords.col(0).maxCoeff());
        minY = std::min(minY, coords.col(1).minCoeff());
        maxY = std::max(maxY, coords.col(1).maxCoeff());
    }

    return {minX, minY, maxX, maxY};
}

void draw(const UtilityFunctionPosterior & upost, std::tuple<Float, Float, Float, Float> boundaries) {
    auto [minX, minY, maxX, maxY] = boundaries;
    const Vector & w = upost.getMean();
    std::cout << "Current weight is: " << w.transpose() << '\n';
    std::cout << "Current cov is:\n" << upost.getCov() << '\n';

    {
        Matrix2D var = upost.getCov().array().square();

        // Transform to larger Cov;
        Vector tr = w.array().exp() / (1.0 + w.array().exp().sum());
        Vector back = (tr.array() / (1.0 - tr.array())).log();
        var.array() /= (w * w.transpose()).array();
        std::cout << "Transformed weight is:\n" << tr.transpose() << " and back " << back.transpose() << '\n';
        std::cout << "Transformed cov is:\n" << var.array().sqrt() << '\n';

        auto coords = plotEllipse(tr, var.array().sqrt());
        // Y = log(X / (1-X))
        // e^Y = X / (1-X)
        // (1-X) * e^Y = X
        // e^Y - X * e^Y = X
        // e^Y = X (1 + e^Y)
        // X = e^Y / (1 + e^Y)

        // Transform back
        Matrix2D ce = coords.array().exp();
        coords = ce.array().colwise() / (1.0f + ce.array().rowwise().sum());

        // Plot
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "-"}});
    }

    std::vector<std::pair<Float, Matrix2D>> lines;
    Matrix2D coords(2, 2); coords.setZero();

    const Float stepN = 5.0;
    const Float step = (maxX - minX) / stepN;
    for (int i = 0; i < stepN + 0.0001; ++i) {
        coords = get2DBoundedPointsFromSlope(-w[1] / w[0], -step * i, 0, 0, maxY - minY);

        const Float d = minX + step * i;
        coords(0, 0) += d;
        coords(0, 1) += minY;
        coords(1, 0) += d;
        coords(1, 1) += minY;

        lines.emplace_back(d, coords);
    }

    for (size_t z = 0; z < lines.size(); ++z) {
        const auto & [d, coords] = lines[z];
        std::string dashes = std::to_string(5) + "," + std::to_string(10 - z * 9.0 / (lines.size() - 1.0));
        plt::plot(toVec(coords.col(0)), toVec(coords.col(1)), {{"linestyle", "dashed"}, {"dashes", dashes}});
    }
}

void draw(const UtilityFunctionParticlePosterior & upost) {
    const auto & particles = upost.getParticles();
    Vector normWeights = upost.getWeights() / upost.getWeights().maxCoeff();

    plt::scatter(toVec(particles.col(0)), toVec(particles.col(1)), toVec(normWeights));
}

void draw(const MOBanditNormalPosterior & bpost, const UtilityFunctionPosterior & upost) {
    plt::clf();

    auto boundaries = draw(bpost);
    draw(upost, boundaries);

    // FIXME: to add again
    // Float score = computeScore(arms, dm, 10000);
    // std::cout << "Current score: " <<  score << '\n';

    plt::pause(0.001);
}

void draw(const MOBanditNormalPosterior & bpost, const UtilityFunctionParticlePosterior & upost) {
    plt::clf();

    draw(bpost);
    draw(upost);

    // FIXME: to add again
    // Float score = computeScore(arms, dm, 10000);
    // std::cout << "Current score: " <<  score << '\n';

    plt::pause(0.001);
}


